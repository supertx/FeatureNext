import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """简单的残差块，用于预测层的深度特征提取"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return F.relu(x + self.conv(x))

class MultiModalQualityNet(nn.Module):
    def __init__(self):
        super(MultiModalQualityNet, self).__init__()
        
        # 1. 输入层：专门针对人脸和掌纹的小卷积分支 (浅层特征提取)
        # 假设输入尺寸为 112x112
        self.face_stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), # 56x56
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 28x28
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.palm_stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # 2. 融合层：拼接后的通道数为 32 + 32 = 64
        # 3. 预测层：共享残差块提取高级质量特征
        self.shared_layers = nn.Sequential(
            ResidualBlock(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 14x14
            ResidualBlock(128),
            nn.AdaptiveAvgPool2d(1) # 全局平均池化 -> 1x1
        )

        # 最终输出单个质量权重（表示 face 的权重），用 sigmoid 将值约束到 [0,1]
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

    def forward(self, face_img, palm_img):
        # 分别提取浅层特征
        feat_face = self.face_stem(face_img)
        feat_palm = self.palm_stem(palm_img)
        
        # 拼接特征 (Channel Dimension)
        merged = torch.cat([feat_face, feat_palm], dim=1)
        
        # 通过共享残差层
        combined_feat = self.shared_layers(merged)
        combined_feat = torch.flatten(combined_feat, 1)
        
        # 得到单个质量权重，代表 face 的权重。palm 的权重可由 (1 - q_face) 获得
        quality_weight = self.fc(combined_feat)
        quality_weight = F.softmax(quality_weight, dim=1)
        return quality_weight[:, 0], quality_weight[:, 1]  # q_face, q_palm

# 测试代码
# model = MultiModalQualityNet()
# q_f, q_p = model(torch.randn(1, 3, 112, 112), torch.randn(1, 3, 112, 112))
# print(f"Face Quality: {q_f.item()}, Palm Quality: {q_p.item()}")