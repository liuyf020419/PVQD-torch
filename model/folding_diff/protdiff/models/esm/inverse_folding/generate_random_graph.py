import os, sys
import torch
import torch.nn.functional as F

from torch_geometric.utils.to_dense_adj import to_dense_adj

def nan_to_num(ts, val=0.0):
    """
    Replaces nans in tensor with a fixed value.    
    """
    val = torch.tensor(val, dtype=ts.dtype, device=ts.device)
    return torch.where(~torch.isfinite(ts), val, ts)


def inv_gumbel_cdf(y, mu=0, beta=1, eps=1e-20):
    return mu - beta * torch.log(-torch.log(y + eps))


def sample_gumbel(shape):
    p = torch.rand(shape)
    return inv_gumbel_cdf(p)


def differentiable_sample_gumbel(logits, temperature=1):
    uniform_noise = torch.rand(logits.shape)
    logits_with_noise = logits - torch.log(-torch.log(uniform_noise))
    return F.softmax(logits_with_noise / temperature)


def distance_propensity(distance, graph_mode='inverse_cubic'):
    if graph_mode == 'inverse_cubic':
        return 1./torch.pow(distance, 3)
    if graph_mode == 'inverse_binary':
        return 1./torch.pow(distance, 2)
    elif graph_mode == 'constant':
        return torch.zeros_like(distance)
    elif graph_mode == 'exponentail':
        return torch.exp(-distance)


def dist_noise(D_adjust, top_k, coord_mask_2D=None, residue_mask_2D=None, Dseq=None, E_idx=None, graph_mode='inverse_cubic', propensity_scale=100):
    bsz, node_num = E_idx.shape[:2]
    D_adjust_prop = distance_propensity(D_adjust, graph_mode)
    # D_adjust_prop = nan_to_num(D_adjust_prop) + (~coord_mask_2D) * (1e8 + Dseq*1e6) + (
    #         ~residue_mask_2D) * (1e10)

    uniform_noise = torch.rand_like(D_adjust_prop)
    D_adjust_noise = D_adjust_prop * propensity_scale - torch.log(-torch.log(uniform_noise))
    D_adjust_noise_adjust = D_adjust_noise + (~coord_mask_2D) * -(1e8 + Dseq*1e6) + (~residue_mask_2D) * -(1e10)
    D_noise_neighbors, E_noise_idx = torch.topk(D_adjust_noise, top_k, dim=-1)

    return E_noise_idx