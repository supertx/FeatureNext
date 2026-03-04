
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
from timm.models.convnext import convnext_tiny

# 假设这些是你项目中已有的模块
from model import get_mbf, PalmNet, ArcfaceHead, MultiModalQualityNet
from dataset import FusionDataset, get_default_transfrom

def process_state_dict(state_dict):
    processed_dict = {}
    for k, v in state_dict.items():
        if k.startswith("backbone"):
            new_key = k.replace("backbone.", "")
            processed_dict[new_key] = v
    return processed_dict

class MultiModalSystem(nn.Module):
    def __init__(self, num_classes, feature_dim=512):
        super(MultiModalSystem, self).__init__()
        # 1. 骨干网络：人脸使用 MobileFaceNet, 掌纹使用自定义 PalmNet
        self.face_net = get_mbf(fp16=False, num_features=feature_dim)
        self.face_net.load_state_dict(torch.load("model/data/model.pt", weights_only=True))

        # self.palm_net = PalmNet(input_dim=3, num_classes=num_classes)
        self.palm_net = convnext_tiny(pretrained=False) # 先不加载 ImageNet 权重
            
        
        # 移除原有的分类头，只做特征提取
        self.palm_net.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(),
            nn.LayerNorm(768, eps=1e-6, elementwise_affine=True),
            nn.Linear(768, feature_dim),
        )


        self.palm_net.load_state_dict(process_state_dict(torch.load("checkpoints/palm_finetune/palm_finetuned_final.pt", weights_only=True)), strict=False)
        
        # 2. 质量评估网络：输入双模态图像，输出 q_f, q_p (Softmax 归一化)
        self.quality_net = MultiModalQualityNet()
        
        # 3. ArcFace Head：用于将特征映射到 ID 空间的 Logits (训练关键)
        # 注意：这里人脸和掌纹共享同一个身份空间
        self.arcface_palm = ArcfaceHead(in_channels=feature_dim, num_classes=num_classes, s=64.0, m=0.5)
        self.arcface_face = ArcfaceHead(in_channels=feature_dim, num_classes=num_classes, s=64.0, m=0.5)

    def forward(self, face_img, palm_img, labels=None):
        # --- A. 独立特征提取 ---
        f_face = self.face_net(face_img)
        f_palm = self.palm_net(palm_img)
        
        # --- B. 特征归一化 (ArcFace 核心要求) ---
        f_face = F.normalize(f_face, p=2, dim=1)
        f_palm = F.normalize(f_palm, p=2, dim=1)
        
        # --- C. 动态质量权重预测 ---
        # q_f, q_p 形状为 [Batch], 且 q_f + q_p = 1
        q_f, q_p = self.quality_net(face_img, palm_img)
        # --- D. 决策逻辑分支 ---
        if labels is not None:
            # 【训练模式】：分值级 Logits 融合
            # 1. 分别计算两个模态独立的 ArcFace Logits
            logits_f = self.arcface_face(f_face, labels)
            logits_p = self.arcface_palm(f_palm, labels)
            
            # 2. 按照质量权重进行线性融合 (Weighted Logit Fusion)
            # 扩展维度以匹配 [Batch, num_classes]
            logits_fusion = q_f.view(-1, 1) * logits_f + q_p.view(-1, 1) * logits_p
            
            return logits_fusion, q_f, q_p
        else:
            # 【推理模式】：返回归一化特征与质量权重
            # 后续匹配时计算：S_total = q_f * cos(f_f, target_f) + q_p * cos(f_p, target_p)
            return f_face, f_palm, q_f, q_p

def fusion_trainer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 19779 
    feature_dim = 512
    
    # 初始化系统
    model = MultiModalSystem(num_classes, feature_dim).to(device)
    
    # 加载第二阶段微调好的权重
    # model.face_net.load_state_dict(torch.load('face_finetuned.pt'), strict=False)
    # model.palm_net.load_state_dict(torch.load('palm_finetuned.pt'), strict=False)

    # 策略：冻结主干网络，仅训练质量网络和分类头
    for param in model.face_net.parameters():
        param.requires_grad = False
    for param in model.palm_net.parameters():
        param.requires_grad = False
    
    optimizer = optim.Adam([
        {'params': model.quality_net.parameters(), 'lr': 1e-4},
        {'params': model.arcface_face.parameters(), 'lr': 1e-3},
        {'params': model.arcface_palm.parameters(), 'lr': 1e-3}
    ], weight_decay=5e-4)

    train_loader = DataLoader(
        FusionDataset("/data/tx/palm_data", "/data/tx/IJB", get_default_transfrom(True)),
        batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )
    
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir="runs/score_level_fusion")
    
    model.train()
    model.face_net.eval() # 保持主干在 eval 模式以固定 BatchNorm
    model.palm_net.eval()

    for epoch in range(40):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", ncols=100)

        for batch_idx, (face_img, palm_img, labels) in enumerate(pbar):
            face_img, palm_img, labels = face_img.to(device), palm_img.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播：得到加权后的融合 Logits
            logits, q_f, q_p = model(face_img, palm_img, labels)
            
            # 计算总损失
            loss = criterion(logits, labels)
            
            predict = torch.argmax(logits, dim=1)
            acc = (predict == labels).float().mean()
            # 可选：熵正则化，防止 QualityNet 变得过于武断 (极端的 0 或 1)
            # entropy_loss = -0.01 * torch.mean(q_f * torch.log(q_f + 1e-6) + q_p * torch.log(q_p + 1e-6))
            # (loss + entropy_loss).backward()
            
            loss.backward()
            optimizer.step()

            # 统计记录
            if batch_idx % 10 == 0:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar("Accuracy/Train", acc.item(), global_step)
                writer.add_scalar("Loss/Total", loss.item(), global_step)
                writer.add_scalar("Quality/Face_Weight", q_f.mean().item(), global_step)
                writer.add_scalar("Quality/Palm_Weight", q_p.mean().item(), global_step)
                
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "q_face": f"{q_f.mean().item():.2f}", "q_palm": f"{q_p.mean().item():.2f}", "acc": f"{acc.item():.4f}"})

        # 每个 Epoch 保存一次质量网络
        torch.save(model.quality_net.state_dict(), f"quality_net_v3_epoch_{epoch}.pth")

    writer.close()

if __name__ == "__main__":
    fusion_trainer()
