import torch.nn as nn
from torchvision import models
import os
import torch
from torchvision.models.video import R2Plus1D_18_Weights
from transformers import TimesformerModel


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class TimeSformer(nn.Module):

    def __init__(self):
        super().__init__()
        self.base_model = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400")

    def forward(self, x):
        x = self.base_model(x)
        x = x.last_hidden_state
        # the output of the timsformer is [B, T, E], with B the batch size, T the number of the sequence tokens and E the embedding dimensions
        # take just the first component of the second dimension, namely the embedding tensor relative to the classification token for all the tensors in the batch
        x = x[:, 0, :]
        return x


class R2plus1d_18(nn.Module):

    def __init__(self):
        super().__init__()
        self.base_model = models.video.r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        self.base_model.fc = Identity()

    def forward(self, x):
        x = self.base_model(x)
        return x


class R3D(nn.Module):

    def __init__(self):
        super().__init__()
        self.base_model = torch.hub.load("facebookresearch/pytorchvideo", "slow_r50", pretrained=True)
        self.base_model.blocks[5].proj = Identity()

    def forward(self, x):
        x = self.base_model(x)
        return x


class R3D_slowfast(nn.Module):

    def __init__(self):
        super().__init__()
        self.base_model = torch.hub.load("facebookresearch/pytorchvideo", "slowfast_r50", pretrained=True)
        self.base_model.blocks[6].proj = Identity()

    def forward(self, x):
        x = self.base_model(x)
        return x