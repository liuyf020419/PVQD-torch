import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import *

import sys
sys.path.append("/train14/superbrain/yfliu25/structure_refine/structure_refine_old_v/models")
from attention.modules import TransformerLayer

import numpy as np

class ActPredBlock(nn.Module):
    def __init__(self, config, single_channel, random_mask=False) -> None:
        super().__init__()
        self.config = config
        self.random_mask = random_mask
        if random_mask:
            self.aatype_embedding = nn.Embedding(22, single_channel)
        
        self.input_layer = Linear(single_channel, self.config.encoder_n_embd, initializer="relu")

        self.act_encoder = nn.ModuleList(
            [
                TransformerLayer(
                    self.config.encoder_n_embd,
                    self.config.encoder_n_embd * 4,
                    self.config.encoder_haeds,
                    dropout = getattr(config, 'dropout', 0.0),
                    add_bias_kv=True,
                    use_esm1b_layer_norm=False,
                )
                for _ in range(self.config.encoder_layers)
            ]
        )
        
        layers=[]
        decoder_channel = config.decoder_channel
        for i, h in enumerate(decoder_channel):
            layers.append(Linear(self.config.encoder_n_embd if i == 0 else decoder_channel[i-1], h, initializer="relu"))
            layers.append(nn.ReLU())

        layers.append(Linear(decoder_channel[-1], self.config.n_output, initializer="zeros"))                
        self.act_decoder= nn.Sequential(*layers)


    def forward(self, act_represent, mask, aatype=None, ca_pos=None):
        # act_represent.shape [B, L, H] or [n, B, L, H]
        if len(act_represent.shape) == 4:
            n, batchsize, L, _ = act_represent.shape
            act_represent = act_represent.reshape(n * batchsize, L, -1)
            init_shapesize = 4
            mask = mask[None].repeat(n, 1, 1).reshape(n * batchsize, L)
        else:
            batchsize, L, _ = act_represent.shape
            init_shapesize = 3

        padding_mask = 1.0 - mask

        if not padding_mask.any():
            padding_mask = None

        if self.random_mask:
            mask_modes = np.random.randint(0, 4, 1)
            aa_mask = gen_random_mask(self.config, batchsize, L, mask_modes[0], ca_pos)
            aatype_masked = torch.where(aa_mask == 0, 21, aatype)
            aa_emb = self.aatype_embedding(aatype_masked)
            act_represent = aa_emb + act_represent

        x = F.relu(self.input_layer(act_represent))
        for layer in self.act_encoder:
            x = x.transpose(0, 1)
            # import pdb; pdb.set_trace()
            x, attn = layer(x, self_attn_padding_mask=padding_mask)
            x = x.transpose(0, 1)
        x = x * mask[..., None]

        x = self.act_decoder(x)

        if init_shapesize == 4:
            return x.reshape(n, batchsize, L, -1)
        else:
            if self.random_mask:
                return x.reshape(batchsize, L, -1), aa_mask
            else:
                return x.reshape(batchsize, L, -1)

        # if init_shapesize == 4:
        #     return x.reshape(n, batchsize, L, -1)
        # else:
        #     return x.reshape(batchsize, L, -1)



def gen_random_mask(config, batchsize, seq_len, mask_mode, ca_pos):
    p_rand = config.p_rand
    p_lin = config.p_lin
    p_spatial = config.p_spatial

    min_lin_len = int(p_lin[0] * seq_len) # 0.25
    max_lin_len = int(p_lin[1] * seq_len) # 0.75
    lin_len = torch.randint(min_lin_len, max_lin_len, [1]).item()

    min_knn = p_spatial[0] # 0.1
    max_knn = p_spatial[1] # 0.5
    knn = int((torch.rand([1]) * (max_knn-min_knn) + min_knn).item() * seq_len)

    if mask_mode == 0: # random 0.5
        mask = (torch.rand(batchsize, seq_len) > p_rand).long()

    elif mask_mode == 1: # linear
        start_index = torch.randint(0, seq_len-lin_len, [batchsize])
        mask = torch.ones(batchsize, seq_len)
        mask_idx = start_index[:, None] + torch.arange(lin_len)
        mask.scatter_(1, mask_idx, torch.zeros_like(mask_idx).float())

    elif mask_mode == 2: # full
        mask = torch.zeros(batchsize, seq_len)

    elif mask_mode == 3: # spatial
        central_absidx = torch.randint(0, seq_len, [batchsize])
        ca_map = torch.mean(ca_pos[:, None] - ca_pos[:, :, None], -1)
        batch_central_knnid = torch.stack([ca_map[bid, central_absidx[bid]] for bid in range(batchsize)])
        knn_idx = torch.argsort(batch_central_knnid)[:, :knn]
        mask = torch.ones(batchsize, seq_len).to(ca_map.device)
        mask.scatter_(1, knn_idx, torch.zeros_like(knn_idx).float())

    return mask.to(ca_pos.device)
