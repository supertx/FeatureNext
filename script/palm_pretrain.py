import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from timm.models.convnext import convnext_tiny, convnext_base
import matplotlib
import cv2 as cv
from tqdm import tqdm


matplotlib.use('Agg')

def warm_up(epoch):
    if epoch <= 10:
        return 0.0001 + (1.0 - 0.1) / (10 - 0) * epoch
    elif 10 < epoch <= 50:
        return 1.0 + (0.0001 - 1.0) / (50 - 10) * (epoch - 10)
    else:
        return 0.0001
# ==========================================
# 1. 核心创新点：Gabor 纹理感知 Loss
# ==========================================
class GaborLoss(nn.Module):
    def __init__(self, device='cuda'):
        super(GaborLoss, self).__init__()
        self.device = device
        self.gabor_kernels = self._create_gabor_kernels().to(device)
        
    def _create_gabor_kernels(self):
        filters = []
        ksize = 31
        sigma = 4.0
        lambd = 10.0
        gamma = 0.5
        psi = 0
        
        # 增加更多方向以捕捉掌纹的复杂纹理 (0, 30, 60, 90, 120, 150)
        thetas = [0, np.pi/6, np.pi/3, np.pi/2, np.pi*2/3, np.pi*5/6]
        
        for theta in thetas:
            y, x = torch.meshgrid(torch.linspace(-(ksize//2), ksize//2, ksize), 
                                  torch.linspace(-(ksize//2), ksize//2, ksize), indexing='ij')
            
            x_theta = x * np.cos(theta) + y * np.sin(theta)
            y_theta = -x * np.sin(theta) + y * np.cos(theta)
            
            gb = torch.exp(-0.5 * (x_theta**2 + gamma**2 * y_theta**2) / sigma**2) * \
                 torch.cos(2 * np.pi * x_theta / lambd + psi)
            filters.append(gb)
            
        # [Out_Channels, In_Channels, H, W]
        # In_Channels = 3 (RGB)
        filters = torch.stack(filters).unsqueeze(1).repeat(1, 3, 1, 1) 
        return filters.float()

    def forward(self, pred, target):
        # padding=15 对应 ksize=31，保持尺寸不变
        pred_feat = F.conv2d(pred, self.gabor_kernels, padding=15)
        target_feat = F.conv2d(target, self.gabor_kernels, padding=15)
        return F.l1_loss(pred_feat, target_feat)

# ==========================================
# 2. FCMAE 模型结构
# ==========================================
class PalmprintFCMAE(nn.Module):
    def __init__(self, mask_ratio=0.6):
        super().__init__()
        self.mask_ratio = mask_ratio
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        # Encoder
        # self.encoder = convnext_tiny(pretrained=True) 
        self.encoder = convnext_tiny(pretrained=True) 
        self.encoder.head = nn.Identity() # 移除分类头
        
        # Decoder
        # ConvNeXt Tiny 输出通道通常是 768
        self.decoder = nn.Sequential(
            nn.Conv2d(768, 384, kernel_size=1),
            nn.PixelShuffle(2), # -> 96 channels, 尺寸 x2
            # 上采样层
            nn.ConvTranspose2d(96, 96, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.ConvTranspose2d(96, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 3, kernel_size=1),
            # 【关键修改】添加 Tanh，将输出限制在 [-1, 1]，匹配输入数据的 Normalize
            nn.Tanh() 
        )

    def random_masking(self, x, mask_ratio):
        """ 像素级随机掩码 """
        B, C, H, W = x.shape
        # 生成 mask: 1 代表保留，0 代表被遮挡
        mask = torch.rand((B, 1, H, W), device=x.device) > mask_ratio
        return x * mask.float(), mask.float()

    def forward(self, x):
        # 1. Masking
        x_masked, mask = self.random_masking(x, self.mask_ratio)
        
        # 2. Encoding
        # ConvNeXt feature map 尺寸通常是输入的 1/32
        latent = self.encoder.forward_features(x_masked)
        
        # 3. Decoding
        pred = self.decoder(latent)
        
        # 4. 尺寸对齐 (防止 ConvNeXt 下采样和 Decoder 上采样不完全匹配)
        if pred.shape[-2:] != x.shape[-2:]:
            pred = F.interpolate(pred, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        return pred, mask

# ==========================================
# 3. 训练函数
# ==========================================
def train_one_epoch(model, dataloader, optimizer, device, epoch_idx):
    model.train()
    mse_criterion = nn.MSELoss()
    gabor_criterion = GaborLoss(device=device)
    
    # 权重策略：随着训练进行，可以逐渐增加纹理 Loss 的权重
    lambda_gabor = 0.2 
    
    total_loss_avg = 0
    
    t = tqdm(dataloader, ncols=100)
    for batch_idx,(_, palm_images, _) in enumerate(t):
        images = palm_images.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        pred_imgs, mask = model(images)
        
        # Loss 计算
        # 1. MSE Loss (像素一致性)
        loss_mse = mse_criterion(pred_imgs, images)
        
        # 2. Gabor Loss (纹理一致性)
        loss_gabor = gabor_criterion(pred_imgs, images)
        
        loss = loss_mse + lambda_gabor * loss_gabor
        
        loss.backward()
        optimizer.step()
        
        total_loss_avg += loss.item()
        t.desc = f"Loss: {loss.item():.4f} (MSE: {loss_mse.item():.4f}, Gabor: {loss_gabor.item():.4f} )"
        
        if batch_idx % 20 == 0:
            save_reconstruction_images(model, dataloader, device, epoch_idx, batch_idx, "vis", n_imgs=4)
    return total_loss_avg / len(dataloader)

import matplotlib.pyplot as plt

def save_reconstruction_images(model, dataloader, device, epoch, step_in_epoch, save_dir, n_imgs=4):
    """
    可视化并保存：原始图 vs 掩码图 vs 重建图
    n_imgs: 要展示几张图片
    """
    model.eval()
    
    # 创建保存目录
    vis_dir = os.path.join(save_dir, "palm")
    os.makedirs(vis_dir, exist_ok=True)
    
    with torch.no_grad():
        # 取一个 Batch 的数据
        # 注意：这里我们只取第一个 Batch 做展示，为了固定观察对象，建议固定一个 Batch
        _, images, _ = next(iter(dataloader))
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
    
    model.train() # 记得切回训练模式！

# ==========================================
# Main
# ==========================================
# 假设 dataset.py 在同一目录下
from dataset import FusionDataset 
from torchvision import transforms as T
from torch.utils.data import DataLoader

def palm_pretrain():
    # 配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128 # 显存不够可调小
    num_epochs = 50
    lr = 2e-3
    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    print(f"Using device: {device}")

    # 【关键修改】Transforms 逻辑修复
    # 先 Resize 确保图够大，再 CenterCrop/RandomCrop 保证核心区域，最后 Normalize
    train_transforms = T.Compose([
            T.ToTensor(),
            # T.RandomCrop((224, 224)),
            T.Resize((112, 112), antialias=False),
            # T.RandomApply(
            #     [T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)],
            #     p=0.5,
            # ),
            # T.RandomRotation(5),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    try:
        dataset = FusionDataset(
            palm_data_dir="/data/tx/palm_data",
            face_data_dir="/data/tx/IJB",
            transform=train_transforms,
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        print(f"Dataset loaded. Total images: {len(dataset)}")
    except Exception as e:
        print(f"数据加载失败，请检查路径: {e}")
        exit()

    # 初始化模型
    model = PalmprintFCMAE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    print("开始训练...")
    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(model, dataloader, optimizer, device, epoch)
        scheduler.step()
        
        print(f"=== Epoch {epoch} finished. Avg Loss: {avg_loss:.4f} ===")
        
        # 每 10 个 Epoch 保存一次
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(save_dir, f"palm_fcmae_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

        # save_reconstruction_images(model, dataloader, device, epoch, save_dir, n_imgs=8)
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(save_dir, f"palm_fcmae_final.pth"))
    print("训练完成！")