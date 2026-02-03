import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class ArcfaceHead(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        s: float = 30.0,
        m: float = 0.5,
    ):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_channels))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, feats: Tensor, labels: Tensor) -> Tensor:
        cosine = F.linear(F.normalize(feats), F.normalize(self.weight))

        cosine = cosine.clamp(-1.0 + 1e-5, 1.0 - 1e-5)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine.float() > self.th, phi.float(),
                          cosine.float() - self.mm)

        one_hot = torch.zeros(cosine.size()).type_as(phi).long()
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


if __name__ == "__main__":
    head = ArcfaceHead(128, 10)
    print(head)
    x = torch.randn(1, 128)
    y = head(x, torch.tensor([0]))
    print(y.shape)
    print(y)
