import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import DataLoader
import math
import os
from timm.models.convnext import convnext_tiny
from dataset.fusion_dataset import FusionDataset, get_default_transfrom

# ==========================================
# 1. 定义 ArcFace Loss (识别任务的核心)
# ==========================================
# ArcFace 能把特征映射到超球面上，极大提升识别准确率
class ArcFaceLayer(nn.Module):
    def __init__(self, in_features, num_classes, s=30.0, m=0.50):
        super(ArcFaceLayer, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s # scale
        self.m = m # margin
        
        # 权重矩阵 [Out, In]
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels):
        # 1. 归一化特征和权重
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        
        # 2. 加上 Margin (核心步骤)
        # 限制 cosine 范围防止 acos 出错
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7) 
        theta = torch.acos(cosine)
        theta_m = theta + self.m
        k = torch.cos(theta_m)
        
        # 3. 构建 One-hot 标签对应的 Logits
        # 只在对应的正确类别上加 Margin，其他类别保持原样
        one_hot = torch.zeros(cosine.size(), device=features.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        logits = one_hot * k + (1.0 - one_hot) * cosine
        logits *= self.s
        
        return logits


# ==========================================
# 2. 定义第二阶段的识别模型
# ==========================================
class PalmprintRecognizer(nn.Module):
    def __init__(self, num_classes, feat_dim=512, pretrained_path=None):
        super().__init__()
        
        # A. 加载骨干网络 (ConvNeXt Tiny)
        self.backbone = convnext_tiny(pretrained=False) # 先不加载 ImageNet 权重
        # ConvNeXt Tiny 输出特征维数通常为 768
        self.feat_dim = feat_dim
        
        
        # 移除原有的分类头，只做特征提取
        self.backbone.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(),
            nn.LayerNorm(768, eps=1e-6, elementwise_affine=True),
            nn.Linear(768, self.feat_dim),
        )
        
        # B. 加载第一阶段的预训练权重 (关键步骤！)
        if pretrained_path:
            self._load_fcmae_weights(pretrained_path)
        
        # C. 添加 ArcFace 层
        self.arcface = ArcFaceLayer(self.feat_dim, num_classes)
        
    def _load_fcmae_weights(self, path):
        """
        从第一阶段的 checkpoint 中只加载 Encoder 部分的权重
        """
        print(f"正在加载第一阶段预训练权重: {path}")
        checkpoint = torch.load(path, map_location='cpu')
        
        # 提取 state_dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict'] # 如果你之前的代码保存了 state_dict 键
        else:
            state_dict = checkpoint # 如果直接保存的是 model.state_dict()
            
        model_dict = self.backbone.state_dict()
        new_state_dict = {}
        
        for k, v in state_dict.items():
            # 第一阶段保存的 key 可能是 "encoder.stages.0..."
            # 我们只需要 "encoder." 开头的部分，并要把 "encoder." 这个前缀去掉
            if k.startswith("encoder."):
                name = k.replace("encoder.", "") # 去掉前缀，匹配 backbone 的 keys
                if name in model_dict:
                    new_state_dict[name] = v
                    
        # 更新权重
        model_dict.update(new_state_dict)
        self.backbone.load_state_dict(model_dict)
        print(f"成功加载了 {len(new_state_dict)} 层权重 (Decoder 被丢弃)")

    def forward(self, x, labels=None):
        # 1. 提取特征
        features = self.backbone(x)
            
        # 2. 如果是训练阶段，经过 ArcFace
        if self.training and labels is not None:
            logits = self.arcface(features, labels)
            return logits
        
        # 3. 如果是测试/推理阶段，直接返回特征向量用于计算余弦相似度
        return F.normalize(features)

# ==========================================
# 3. 第二阶段训练循环
# ==========================================
def train_stage2(model, dataloader, optimizer, device, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss() # ArcFace 输出 logits 后接 CE Loss
    
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (_, images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward (传入 labels 计算 ArcFace Loss)
        logits = model(images, labels)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 计算简单的准确率
        _, predicted = torch.max(logits.data, 1)
        # print(predicted)
        # print(labels)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if batch_idx % 10 == 0:
            print(f"Stage2 Epoch [{epoch}] Step [{batch_idx}] "
                  f"Loss: {loss.item():.4f} Acc: {100 * correct / total:.2f}%")

def palm_finetune():
    # 配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 第一阶段保存的模型路径
    stage1_weight_path = "./checkpoints/palm_pretrain/palm_fcmae_final.pth" 
    batch_size = 128
    train_transforms = T.Compose([
            T.ToTensor(),
            # T.RandomCrop((224, 224)),
            T.Resize((112, 112), antialias=False),
          
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    save_dir = os.path.join("checkpoints", "palm_finetune")
    # 初始化模型
    dataset = FusionDataset(
        palm_data_dir="/data/tx/palm_data",
        face_data_dir="/data/tx/IJB",
        transform=train_transforms,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    num_classes = dataset.get_class_num()

    model = PalmprintRecognizer(num_classes=num_classes, pretrained_path=stage1_weight_path).to(device)
    
    # 优化器
    # 注意：ArcFace 的参数通常需要稍微大一点的学习率，或者和 backbone 一样
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # 假设 dataloader 已经准备好，并且会返回 (image, label_id)
    
    print("开始第二阶段微调 (Fine-tuning)...")
    # 模拟训练
    for epoch in range(20):
        train_stage2(model, dataloader, optimizer, device, epoch)
        # 保存权重
        torch.save(model.state_dict(), os.path.join(save_dir, f"palm_finetuned_epoch_{epoch+1}.pt"))
    torch.save(model.state_dict(), os.path.join(save_dir, f"palm_finetuned_final.pt"))
        # scheduler.step()