import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class ProteinMPNNEmbedding(nn.Module):
    def __init__(self, config, global_config, down_sampling_scale) -> None:
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.embed_dim = self.config.encoder_embed_dim

        self.single_act = nn.Linear(3, self.config.encoder_embed_dim)


    def forward(self, batch):
        embd = self.single_act(batch['gt_pos'][..., 1, :])
        return embd