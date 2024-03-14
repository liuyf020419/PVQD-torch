import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class ESMConditioner(nn.Module):
    def __init__(self, config, global_config) -> None:
        super().__init__()
        self.config = config
        self.global_config = global_config

        self.esm_rep_dim = self.config.esm_rep_dim
        self.out_rep_dim = self.embed_dim = self.config.out_rep_dim
        self.projection_mode = self.config.projection_mode

        if (self.projection_mode == 'linear'):
            self.activate_rep = nn.Linear(self.esm_rep_dim, self.out_rep_dim)
        elif (self.projection_mode == 'MLP'):
            self.activate_rep = nn.Sequential(
                nn.Linear(self.esm_rep_dim, self.out_rep_dim),
                nn.GELU(),
                nn.Linear(self.out_rep_dim, self.out_rep_dim),
            )

    def forward(self, batch):
        condition_mask = batch['condition_mask']
        esm_rep = batch['esm_rep']
        act_esm_rep = self.activate_rep(esm_rep)
        return condition_mask[..., None] * act_esm_rep


    