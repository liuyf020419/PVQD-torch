import string
import sys
sys.path.append('/train14/superbrain/yfliu25/ProteinMPNN/vanilla_proteinmpnn')

import torch
import torch.nn.functional as F

from ..nn_utils import fape_loss, fape_loss_multichain
from ..protein_utils.backbone import affine_to_coord, coord_to_frame_affine


mpnnalphabet = 'ACDEFGHIKLMNPQRSTVWYX'
chainalphabet = string.ascii_uppercase


def enable_grad(data: torch.Tensor):
    data_in = data.detach().requires_grad_(True)
    return data_in
    

def batch_select(batch_data, select, stack=True):
    unstacked = [b_data[select[b_idx]] for b_idx, b_data in enumerate(batch_data)]
    if stack:
        return torch.stack(unstacked)
    else:
        return unstacked


def selected_nll(log_probs, target_S, selected):
    selected_log_probs = batch_select(log_probs, selected)
    selected_target_S = batch_select(target_S, selected)
    selected_nll_loss = F.nll_loss(
        selected_log_probs.contiguous().view(-1,selected_log_probs.size(-1)), 
        selected_target_S.contiguous().view(-1), reduction='none').view(selected_target_S.size())

    return selected_nll_loss


def mpnn_gradient(X, S_stacked, residue_idx, chain_idx, model, fixbb_mode: str, selected=None):
    device = model.device
    batch_size, L = X.shape[:2]
    if selected is None:
        selected = torch.ones((batch_size, L)).bool().to(device)
    if len(selected.shape) == 1:
        selected = selected[None]
    assert len(selected.shape) == 2
    assert isinstance(S_stacked[0], str)

    X = X.to(device)
    seq_encoded = torch.stack([
        torch.tensor([mpnnalphabet.index(aa) 
        for aa in seq_str]).long() 
        for seq_str in S_stacked]).to(device)

    mask = torch.ones((batch_size, L)).to(device)
    chain_M = torch.ones((batch_size, L)).to(device)
    chain_M_pos = torch.ones((batch_size, L)).to(device)
    randn_1 = torch.randn(chain_M.shape, device=device)
    
    with torch.enable_grad():
        X = enable_grad(X)
        site_log_probs = model(
            X, seq_encoded, mask, chain_M*chain_M_pos, 
            residue_idx, chain_idx, randn_1
            ) # B, L, 20

        site_prob = F.softmax(torch.exp(site_log_probs), dim=-1)
        site_entropy = list( map( lambda b_site_prob: torch.stack(
                [torch.distributions.Categorical(probs=b_site_prob[i]).entropy() \
                for i in range(L)], dim=0) , site_prob) )
        selected_site_entropy = batch_select(torch.stack(site_entropy), selected)

        selected_nll_loss = selected_nll(site_log_probs, S, selected)

        selected_logp = batch_select(torch.max(site_log_probs, dim=-1)[0], selected)

        if fixbb_mode == 'selected_S':
            gradient = torch.autograd.grad(-selected_nll_loss.sum(), X)[0]
        elif fixbb_mode == 'selected_entropy':
            gradient = torch.autograd.grad(-selected_site_entropy.sum(), X)[0]
        elif fixbb_mode == 'selected_logp':
            gradient = torch.autograd.grad(selected_logp.sum(), X)[0]
        else:
            raise ValueError(f'guide mode: {fixbb_mode} not availbale')

    return gradient


def rmsd_gradient(pred_hidden, pred_idx, target_ca_coords, target_idx, batch, decoder_fn, ):
    assert (pred_idx.shape == target_idx.shape)
    gradient_list = []
    batchsize = pred_hidden.shape[0]
    traj_rmsd = []
    for batch_idx in range(batchsize):
        cur_pred_hidden = pred_hidden[batch_idx][None]
        with torch.enable_grad():
            cur_pred_hidden = enable_grad(cur_pred_hidden)
            if not cur_pred_hidden.grad is None:
                cur_pred_hidden.grad.zero_()
            pred_ca_coords = decoder_fn(cur_pred_hidden, batch)[0][..., 4:]
            pred_motif_coords = pred_ca_coords[pred_idx]
            target_motif_coords = target_ca_coords[target_idx]
            motif_rmsd = -compute_rmsd(target_motif_coords, pred_motif_coords)

            motif_rmsd.backward()
            gradient = cur_pred_hidden.grad
            gradient_list.append(gradient)
            traj_rmsd.append(-motif_rmsd.item())

    return torch.cat(gradient_list).detach(), torch.tensor(traj_rmsd)


def compute_rmsd(true_atom_pos, pred_atom_pos, atom_mask=None, eps=1e-6):        
    sq_diff = torch.square(true_atom_pos - pred_atom_pos).sum(dim=-1, keepdim=False) # B, L
    if atom_mask is not None:
        sq_diff = sq_diff[atom_mask]
    msd = torch.mean(sq_diff, -1) # B,
    msd = torch.nan_to_num(msd, nan=1e8)
    return torch.sqrt(msd + eps)


def fape_gradient(pred_hidden, pred_idx, target_coords, target_idx, batch, decoder_fn, fape_config):
    assert (pred_idx.shape == target_idx.shape)
    gradient_list = []
    traj_fape = []
    batchsize, resnum = pred_hidden.shape[:2]
    for batch_idx in range(batchsize):
        cur_pred_hidden = pred_hidden[batch_idx][None]
        with torch.enable_grad():
            cur_pred_hidden = enable_grad(cur_pred_hidden)
            if not cur_pred_hidden.grad is None:
                cur_pred_hidden.grad.zero_()
            pred_affine = decoder_fn(cur_pred_hidden, batch)[0]

            pred_motif_affine = pred_affine[pred_idx][None]
            pred_motif_coords = affine_to_coord(pred_motif_affine)
            target_motif_coords = target_coords[target_idx][None]
            target_motif_affine = coord_to_frame_affine(target_motif_coords)['affine']
            motif_fape = -fape_loss_multichain(pred_motif_affine[None], target_motif_affine, target_motif_coords[..., :3, :], batch['single_mask'][0][None], batch['chain_idx'][0][None], fape_config,)[0]['fape_loss']

            motif_fape.backward()
            gradient = cur_pred_hidden.grad
            gradient_list.append(gradient)
            traj_fape.append(-motif_fape.item())

    return torch.cat(gradient_list).detach(), torch.tensor(traj_fape)


def update_x_with_grad(x, guide_scale, gradient):
    x = x + guide_scale * gradient
    return x

