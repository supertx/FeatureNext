import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import get_mbf, PalmNet, MultiModalQualityNet, ArcfaceHead
from dataset import FusionDataset, get_default_transfrom


class MultiModalSystem(nn.Module):
    def __init__(self, num_classes, feature_dim=512):
        super(MultiModalSystem, self).__init__()
        # 1. 加载主干网络
        self.face_net = get_mbf(fp16=False, num_features=feature_dim)
        self.palm_net = PalmNet(input_dim=3, num_classes=num_classes, Weight_allocator=None)
        
        # 2. 质量评估网络
        self.quality_net = MultiModalQualityNet()
        
        # 3. ArcFace Head (用于训练阶段)
        self.arcface = ArcfaceHead(in_channels=feature_dim, num_classes=num_classes, s=64.0, m=0.5)

    def forward(self, face_img, palm_img, labels=None):
        # 提取特征
        f_face = self.face_net(face_img)
        f_palm = self.palm_net.extract_feature(palm_img)
        
        # 特征归一化 (ArcFace 核心要求)
        f_face = torch.nn.functional.normalize(f_face, p=2, dim=1)
        f_palm = torch.nn.functional.normalize(f_palm, p=2, dim=1)
        
        # 获取质量分
        q_face, q_palm = self.quality_net(face_img, palm_img)
        
        # 质量加权融合
        # unsqueeze(1) 将 [B] 变为 [B, 1] 以便进行广播乘法
        f_fusion = q_face.unsqueeze(1) * f_face + q_palm.unsqueeze(1) * f_palm
        
        # 如果提供了 labels，说明在训练模式，返回 ArcFace 输出
        if labels is not None:
            return self.arcface(f_fusion, labels), q_face, q_palm
        
        # 否则返回融合特征用于推理
        return f_fusion

def fusion_trainer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 19779 # 需与数据集匹配
    feature_dim = 512
    
    # 1. 初始化封装系统
    model = MultiModalSystem(num_classes, feature_dim).to(device)
    
    # 2. 加载预训练权重 (如果有)
    model.face_net.load_state_dict(torch.load('/home/power/tx/FeatureNext/model/model_data/student_54_epoch.pt'))
    # model.palm_net.backbone.load_state_dict(torch.load('/home/power/tx/FeatureNext/model/model_data/best_model.pt'))

    # 3. 训练策略设置
    # 冻结两个主干网络，只训练 QualityNet 和 ArcFace 的权重
    for param in model.face_net.parameters():
        param.requires_grad = False
    for param in model.palm_net.parameters():
        param.requires_grad = False
    
    # 仅优化质量网络和 ArcFace Head 的参数
    optimizer = optim.Adam([
        {'params': model.quality_net.parameters()},
        {'params': model.arcface.parameters()}
    ], lr=1e-4)

    # 4. 数据
    train_loader = DataLoader(
        FusionDataset("/data/tx/palm_data", "/data/tx/IJB", get_default_transfrom(True)),
        batch_size=32, shuffle=True, num_workers=4
    )
    
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir="runs/fusion_train")
    global_step = 0

    # 5. 训练循环
    model.train()
    model.face_net.eval() # 主干保持 eval 模式以稳定 BN
    model.palm_net.eval()

    for epoch in range(20):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for face_img, palm_img, labels in pbar:
            face_img, palm_img, labels = face_img.to(device), palm_img.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播得到 ArcFace Logits
            logits, q_f, q_p = model(face_img, palm_img, labels)
            # 计算 ArcFace 损失
            loss = criterion(logits, labels)
            
            # 可选：正则化项，鼓励质量分不要过于两极分化 (Entropy Regularization)
            # loss += -0.01 * torch.mean(q_f * torch.log(q_f + 1e-6) + q_p * torch.log(q_p + 1e-6))

            loss.backward()
            optimizer.step()

            # TensorBoard 记录
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/q_face", q_f.mean().item(), global_step)
            writer.add_scalar("train/q_palm", q_p.mean().item(), global_step)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "q_face": f"{q_f.mean().item():.2f}"})
            global_step += 1

        # 保存质量网络权重
        torch.save(model.quality_net.state_dict(), f"quality_net_arcface_{epoch}.pth")
        writer.flush()

    writer.close()

