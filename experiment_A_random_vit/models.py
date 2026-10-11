import torch.nn as nn
from torchvision.models import vit_b_16


class RandomInitViT(nn.Module):

    def __init__(self, num_classes=2):

        super().__init__()

        # ==================================================
        # ViT-B/16
        #
        # weights=None
        #
        # 非常重要：
        # 不使用 ImageNet 预训练
        # 不使用 MAE 预训练
        # 完全随机初始化
        # ==================================================

        self.vit = vit_b_16(
            weights=None
        )

        # ==================================================
        # torchvision ViT 默认分类头输出1000类
        #
        # 猫狗只有2类
        # ==================================================

        in_features = self.vit.heads.head.in_features

        self.vit.heads.head = nn.Linear(
            in_features,
            num_classes
        )

    def forward(self, x):

        return self.vit(x)

