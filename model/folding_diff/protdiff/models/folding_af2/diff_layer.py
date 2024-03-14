from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import *


class ScaleVec(nn.Module):
    """Scale vec up or down"""
    def __init__(self, scale_factor, direction: str):
        super().__init__()
        assert direction in ["Down", "Up"]
        self.direction = direction
        self.scale_factor = scale_factor

    def forward(self, affine_t):
        if self.direction == "Up":
            new_affine_t = affine_t * self.scale_factor
        elif self.direction == "Down":
            new_affine_t = affine_t / self.scale_factor

        return new_affine_t


class NoiseMerge(nn.Module):
    """
    Merge noise layer for diffusion process
    We assume ch_dim for pair input are in the last fim
    """
    def __init__(self, noise_ch, act_noise_ch, non_linear="relu", batch_input=False):
        super().__init__()
        self.temb_layer = Linear(noise_ch, act_noise_ch, initializer="relu")
        self.batch_input = batch_input

        if non_linear == "relu":
            self.nonlntemb = nn.ReLU()
        elif non_linear == "silu":
            self.nonlntemb = nn.SiLU()

    def forward(self, repre):
        input_nd = repre["input_nd"]
        temb = repre["temb"]
        temb_ = self.temb_layer(self.nonlntemb(temb))
        # print(input_nd.shape)
        # print(temb_.shape)
        if len(temb_.shape) == len(input_nd.shape):
            input_nd_ = input_nd + temb_
        else:
            if len(input_nd.shape)==3:
                if self.batch_input:
                    input_nd_ = input_nd + temb_[:, None, :]
                else:
                    input_nd_ = input_nd + temb_[None, :, :]

            elif len(input_nd.shape)==4:
                input_nd_ = input_nd + temb_[:, None, None, :]

        return input_nd_
                
            
class IPAnormOut(nn.Module):
    def __init__(self, scale_factor=100, direction="Down", non_linear="relu"):
        super().__init__()
        assert direction == "Down"
        self.scale_vec_down = ScaleVec(scale_factor=scale_factor, direction=direction)
        
        if non_linear == "relu":
            self.quat_nonln = nn.ReLU()
        elif non_linear == "silu":
            self.quat_nonln = nn.SiLU()

        self.quat_emb_layer = Linear(4, 4)
    def forward(self, affine):
        affine_ch = affine.to_tensor()
        quat_ = self.quat_nonln(self.quat_emb_layer(affine_ch[:, :, :4]))
        trans_ = self.scale_vec_down(affine_ch[:, :, 4:])

        return torch.cat([quat_, trans_], -1)
        
