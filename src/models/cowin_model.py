import torch
import torch.nn as nn
from timm import create_model


class CowinBMSModel(nn.Module):
    def __init__(self, num_classes=5, backbone_name="efficientnet_b0"):
        super().__init__()
        self.backbone = create_model(
            backbone_name,
            pretrained=False,
            num_classes=0,
            global_pool=""
        )
        backbone_features = 1280
        self.mi_head = nn.Sequential(
            nn.Linear(backbone_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        self.grade_head = nn.Linear(backbone_features, num_classes)

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        if feats.dim() > 2:
            feats = feats.mean(dim=[2, 3])
        mi = self.mi_head(feats).squeeze(1)
        logits = self.grade_head(feats)
        return mi, logits

