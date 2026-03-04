import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import mxnet as mx
import cv2 as cv
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from timm.models.convnext import convnext_tiny, convnext_base
from dataset import MXFaceDataset
from matplotlib import pyplot as plt
import matplotlib
from tqdm import tqdm
matplotlib.use('Agg')


def warmup_lr(epoch: int):
    if epoch <= 1:
        return 0.1 
    elif 1 < epoch <= 5:
        return 1.0 + (0.1 - 1.0) / (1 - 10) * (epoch - 10)
    

# ==========================================
# 3. 模型结构：FaceFCMAE (ConvNeXt V2 for Face)
# ==========================================
class FaceFCMAE(nn.Module):
    def __init__(self, mask_ratio=0.6):
        super().__init__()
        self.mask_ratio = mask_ratio
        
        # Encoder: ConvNeXt Tiny
        self.encoder = convnext_tiny(pretrained=False) 
        self.encoder.head = nn.Identity() 
        
        # Decoder: 针对 112x112 输入设计的解码器
        # Encoder 输出尺寸通常是 1/32 -> 3x3 (如果输入112) 或者 4x4 (如果输入128)
        # 这里假设输入是 112x112，经过 convnext_tiny 得到 [B, 768, 3, 3]
        
        self.decoder = nn.Sequential(
            nn.Conv2d(768, 384, kernel_size=1),
            
            # Upsample 1: 3x3 -> 7x7
            nn.ConvTranspose2d(384, 192, kernel_size=3, stride=2, padding=0), 
            nn.BatchNorm2d(192), nn.GELU(),
            
            # Upsample 2: 7x7 -> 14x14
            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(96), nn.GELU(),
            
            # Upsample 3: 14x14 -> 28x28
            nn.ConvTranspose2d(96, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.GELU(),
            
            # Upsample 4: 28x28 -> 56x56
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.GELU(),
            
            # Upsample 5: 56x56 -> 112x112
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # 归一化到 [-1, 1]
        )

    def random_masking(self, x, mask_ratio):
        B, C, H, W = x.shape
        mask = torch.rand((B, 1, H, W), device=x.device) > mask_ratio
        return x * mask.float(), mask.float()

    def forward(self, x):
        # 1. Masking
        x_masked, mask = self.random_masking(x, self.mask_ratio)
        
        # 2. Encoding
        latent = self.encoder.forward_features(x_masked)
        print(latent.shape)
        # 3. Decoding
        pred = self.decoder(latent)
        
        # 4. 强制对齐尺寸 (防止 padding 计算误差)
        if pred.shape[-2:] != x.shape[-2:]:
            pred = F.interpolate(pred, size=x.shape[-2:], mode='bilinear')
            
        return pred, mask

# ==========================================
# 4. 训练循环 (Training Loop)
# ==========================================
def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    mse_criterion = nn.MSELoss()
    # gabor_criterion = GaborLoss(device=device)
    
    # 权重策略
    lambda_gabor = 0.1
    
    total_loss = 0
    t = tqdm(dataloader, ncols=60)

    for batch_idx, (images, _) in enumerate(t):
        images = images.to(device)
        
        optimizer.zero_grad()
        pred_imgs, mask = model(images)
        
        loss_mse = mse_criterion(pred_imgs, images)
        # loss_gabor = gabor_criterion(pred_imgs, images)
        
        loss = loss_mse 
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        t.desc = f"Loss: {loss.item():.4f} (MSE: {loss_mse.item():.4f})"
        if batch_idx % 2000 == 0:
            save_reconstruction_images(model, dataloader, device, epoch, batch_idx, "vis", n_imgs=4)

    
def save_reconstruction_images(model, dataloader, device, epoch, step_in_epoch, save_dir, n_imgs=4):
    """
    可视化并保存：原始图 vs 掩码图 vs 重建图
    n_imgs: 要展示几张图片
    """
    model.eval()
    
    # 创建保存目录
    vis_dir = os.path.join(save_dir, "face")
    os.makedirs(vis_dir, exist_ok=True)
    
    with torch.no_grad():
        # 取一个 Batch 的数据
        # 注意：这里我们只取第一个 Batch 做展示，为了固定观察对象，建议固定一个 Batch
        images, _ = next(iter(dataloader))
        images = images.to(device)[:n_imgs] # 只取前 n_imgs 张
        
        # 模型推理
        pred_imgs, mask = model(images)
        
        # --- 数据反归一化 (De-normalization) ---
        # 训练时用了 Normalize(0.5, 0.5)，这里要还原回 [0, 1] 才能显示
        # 形状变换: [B, C, H, W] -> [B, H, W, C]
        
        def denorm(img_tensor):
            return img_tensor * 0.5 + 0.5

        # 1. 原始图
        orig_imgs = denorm(images).cpu().permute(0, 2, 3, 1).clamp(0, 1).numpy()
        temp = cv.cvtColor((orig_imgs[0] * 255).astype(np.uint8), cv.COLOR_RGB2BGR)
        cv.imwrite(os.path.join(vis_dir, f"orig_{epoch}_{step_in_epoch}.png"), temp)
        # 2. 掩码图 (模型看到的残缺图)
        # mask: 1=保留(Visible), 0=被遮挡(Masked)
        # 对应位置相乘，被遮挡的地方变黑
        masked_inputs = denorm(images * mask).cpu().permute(0, 2, 3, 1).clamp(0, 1).numpy()
        temp = cv.cvtColor((masked_inputs[0] * 255).astype(np.uint8), cv.COLOR_RGB2BGR)
        cv.imwrite(os.path.join(vis_dir, f"masked_{epoch}_{step_in_epoch}.png"), temp)
        
        # 3. 重建图
        # FCMAE 的 pred 通常是全图重建，或者我们可以把原图未遮挡的部分贴回去混合显示
        # 这里直接展示纯粹的重建结果，看模型修补得怎么
        recon_imgs = denorm(pred_imgs).cpu().permute(0, 2, 3, 1).clamp(0, 1).numpy()
        temp = cv.cvtColor((recon_imgs[0] * 255).astype(np.uint8), cv.COLOR_RGB2BGR)
        cv.imwrite(os.path.join(vis_dir, f"recon_{epoch}_{step_in_epoch}.png"), temp)
        
    # --- 绘图 ---
    fig, axs = plt.subplots(3, n_imgs, figsize=(n_imgs * 3, 9))
    # 如果 n_imgs=1，axs 需要增加维度
    if n_imgs == 1: axs = axs[:, None]

    for i in range(n_imgs):
        # 第一行：原始图
        axs[0, i].imshow(orig_imgs[i])
        axs[0, i].set_title("Original")
        axs[0, i].axis('off')
        
        # 第二行：输入给模型的掩码图
        axs[1, i].imshow(masked_inputs[i])
        axs[1, i].set_title("Masked Input")
        axs[1, i].axis('off')
        
        # 第三行：模型重建结果
        axs[2, i].imshow(recon_imgs[i])
        axs[2, i].set_title("Reconstruction")
        axs[2, i].axis('off')
        
    plt.tight_layout()
    save_path = os.path.join(vis_dir, f"epoch_{epoch}_{step_in_epoch}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"可视化图片已保存: {save_path}")
    


from torchvision import transforms
def face_pretrain():
    # 配置参数
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64  # 根据显存调整
    lr = 1e-4
    epochs = 5
    
    # 数据集路径 (修改为你自己的 .rec 文件夹路径)
    root_dir = "/data/tx/MS1MV3" 
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    if not os.path.exists(root_dir):
        print(f"Warning: 路径 {root_dir} 不存在，请修改代码中的 root_dir")
        # 这里为了演示方便，不阻断运行，但在实际使用中请确保路径正确
    else:
        try:
            dataset = MXFaceDataset(root_dir, 
                                    transform=transform,)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
            print(f"数据加载成功，共 {len(dataset)} 张图片")
            
            # 初始化模型
            model = FaceFCMAE(mask_ratio=0.6).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
            lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=warmup_lr)
            
            print("开始人脸 FCMAE 预训练...")
            for epoch in range(epochs):
                train_one_epoch(model, dataloader, optimizer, device, epoch)
                
                # 保存权重
                torch.save(model.state_dict(), f"face_fcmae_epoch_{epoch+1}.pth")
                
        except Exception as e:
            e.with_traceback()
            print(f"Dataset加载出错: {e}")