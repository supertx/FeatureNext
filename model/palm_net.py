import torch
from torch import nn, Tensor
import torch.nn.functional as F
import os
import sys
from einops import rearrange

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))
from timm.models import create_model


class ChannelAttention(nn.Module):

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.avg_pool(x).view(b, c)
        max_ = self.max_pool(x).view(b, c)

        avg = self.fc(avg).view(b, c, 1, 1)
        max_ = self.fc(max_).view(b, c, 1, 1)

        out = avg + max_
        out = self.dropout(out)
        return x * out


class SpatialAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)

        attention = self.conv(combined)
        attention = self.sigmoid(attention)
        attention = self.dropout(attention)
        return x * attention


class CBAM(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.channel_att = ChannelAttention(channels)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

class RoutingGate(nn.Module):
    def __init__(self, in_channels, num_experts=3, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, num_experts),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        pooled = self.avg_pool(x).view(b, c)
        weights = self.fc(pooled)  # [B, num_experts]
        weights = torch.softmax(weights, dim=1)
        return weights  # 每个batch的专家权重

class LayerNorm2d(nn.Module):
    """
    针对 (B, C, H, W) 输入的 LayerNorm，模仿 ConvNeXt 的风格。
    它将输入 permute 到 (B, H, W, C) 进行归一化，然后再转回来。
    """
    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.normalized_shape = (num_channels,)
        self.eps = eps
        self.elementwise_affine = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, H, W, C]
        u = x.permute(0, 2, 3, 1)
        # Apply LayerNorm
        u = F.layer_norm(u, self.normalized_shape, self.weight, self.bias, self.eps)
        # [B, H, W, C] -> [B, C, H, W]
        x = u.permute(0, 3, 1, 2)
        return x


class MultiScaleResBlock(nn.Module):

    def __init__(self, channels, use_layernorm=True):
        super().__init__()
        def get_norm(dim):
            if use_layernorm:
                return LayerNorm2d(dim)
            else:
                return nn.BatchNorm2d(dim)

        self.branch1 = nn.Sequential(
            nn.Conv2d(channels,
                      channels,
                      3,
                      padding=1,
                      groups=channels,
                      bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            get_norm(channels),
            nn.GELU(),
            nn.Dropout2d(0.1),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(channels,
                      channels,
                      5,
                      padding=2,
                      groups=channels,
                      bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            get_norm(channels),
            nn.GELU(),
            nn.Dropout2d(0.1),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(channels,
                      channels,
                      7,
                      padding=3,
                      groups=channels,
                      bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            get_norm(channels),
            nn.GELU(),
            nn.Dropout2d(0.1),
        )

        self.routing = RoutingGate(channels, num_experts=3)

        self.attention = CBAM(channels * 3)
        self.fuse = nn.Conv2d(channels * 3, channels, 1)

    def forward(self, x):
        residual = x
        B = x.size(0)

        weights = self.routing(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        b1 = b1 * weights[:, 0].view(B, 1, 1, 1)
        b2 = b2 * weights[:, 1].view(B, 1, 1, 1)
        b3 = b3 * weights[:, 2].view(B, 1, 1, 1)

        combined = torch.cat([b1, b2, b3], dim=1)
        attended = self.attention(combined)
        fused = self.fuse(attended)

        return fused + residual


class Backbone(nn.Module):

    def __init__(self, input_dim=3, feature_dim=512, scale=1, use_layernorm=True):
        super().__init__()
        # 初始大核卷积层
        self.scale = scale
        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, 32 * self.scale, 5, stride=2, padding=2, bias=False),
            LayerNorm2d(32 * self.scale) if use_layernorm else nn.BatchNorm2d(32 * self.scale),
            nn.GELU(),
            # nn.MaxPool2d(3, stride=2, padding=1),
            nn.Dropout2d(0.1),
        )
        # 残差阶段
        self.stage1 = self._make_stage(32 * self.scale, 64 * self.scale, 4)
        self.transition01 = self._make_transition(32 * self.scale, 64 * self.scale, 2)
        self.stage2 = self._make_stage(64 * self.scale, 128 * self.scale, 6)
        self.transition02 = self._make_transition(32 * self.scale, 128 * self.scale, 4)
        self.transition12 = self._make_transition(64 * self.scale, 128 * self.scale, 2)
        self.stage3 = self._make_stage(128 * self.scale, 256 * self.scale, 3)

        # 分类头
        self.head = nn.Sequential(
            nn.Conv2d(256* self.scale, 256* self.scale, kernel_size=7, stride=1, padding=0),
            nn.Flatten(),
            nn.Linear(256* self.scale, feature_dim),
        )

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_transition(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def _make_stage(self, in_channels, out_channels, blocks):
        layers = []
        # 下采样块
        layers.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ))
        # 多尺度残差块
        for _ in range(blocks):
            layers.append(MultiScaleResBlock(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.stem(x)  # [B, 32, 32, 32]
        x1 = self.stage1(x0)  # [B, 64, 16, 16]
        x2 = self.stage2(x1 + self.transition01(x0))  # [B, 128, 8, 8]
        x3 = self.stage3(x2 + self.transition02(x0) +
                         self.transition12(x1))  # [B, 256, 4, 4]
        out = self.head(x3)
        return out



class PalmNet(torch.nn.Module):

    def __init__(
        self,
        input_dim,
        num_classes: int,
        Weight_allocator,
    ):
        super().__init__()

        hidden_dim = 512
        self.num_classes = num_classes
        self.backbone = Backbone(input_dim, hidden_dim, 2)
        

    def get_backbone_parameters(self):
        return self.backbone.parameters()

    def get_head_parameters(self):
        return self.cls_head.parameters()
    
    def extract_feature(self, rois: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(rois)
        feat = F.normalize(feat, p=2, dim=-1, eps=1e-6)
        return feat

    def forward(self, rois: torch.Tensor) -> torch.Tensor:
        return self.extract_feature(rois)


if __name__ == "__main__":
    import torch
    from torchsummary import summary

    model = PalmNet(
        num_classes=1000,
        Weight_allocator=None,
    )
    data = torch.randn(1, 3, 128, 128)
    model.eval()
    model.cuda()
    summary(model, (3, 128, 128))
    # model(data.cuda()) --- IGNORE ---