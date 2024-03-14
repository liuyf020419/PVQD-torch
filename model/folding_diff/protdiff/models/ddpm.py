import logging
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .protein_utils.rigid import affine6_to_affine7, affine7_to_affine6
from .protein_utils.backbone import coord_to_frame_affine
from .protein_utils.add_o_atoms import add_atom_O, batch_add_atom_O, batch_add_atom_O_new
from .protein_utils.write_pdb import write_multichain_from_atoms
from .folding_af2 import r3
from .nn_utils import make_mask, mask_loss, latent_loss, fape_loss_multichain, l2_distance_loss, get_coords_dict_from_affine, nll_loss, sc_simi_loss
from .latent_diff_model import LatentDiffModel
from .conditioner.gvp_conditioner import GVPConditioner
from .conditioner.esm_conditioner import ESMConditioner
# from .latent_diff_model_self_cond import LatentDiffModel as LatentDiffModelSelfCond
from .external_guide.fixbb_score_guide_utils import rmsd_gradient, update_xt_with_grad, fape_gradient

logger = logging.getLogger(__name__)


def loss_dict_fp32(dict:dict):
    fp32_dict = {}
    for k, v in dict.items():
        fp32_dict[k] = v.float()

    return fp32_dict


def guide_scale_scheme(scheme, T):
    decay_types = {
        'constant': lambda t: 1,
        'linear'  : lambda t: t/T,
        'quadratic' : lambda t: t**2/T**2,
        'cubic' : lambda t: t**3/T**3 
    }
    return decay_types[scheme]


class DDPM(nn.Module):
    def __init__(self, config, global_config) -> None:
        super().__init__()
        self.config = config
        self.global_config = global_config

        beta_start, beta_end = global_config.diffusion.betas
        T = global_config.diffusion.T
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

        # Calculations for posterior q(y_{t-1} | y_t, y_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_variance = torch.cat([posterior_variance[1][None], posterior_variance[1:]])

        posterior_log_variance_clipped = posterior_variance.log()
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        posterior_mean_coef1 = betas * alphas_cumprod_prev.sqrt() / (1 - alphas_cumprod)
        posterior_mean_coef2 = (1 - alphas_cumprod_prev) * alphas.sqrt() / (1 - alphas_cumprod)
        # posterior_mean_coef3 = (1 - (betas * alphas_cumprod_prev.sqrt() + alphas.sqrt() * (1 - alphas_cumprod_prev))/ (1 - alphas_cumprod)) # only for mu from prior
        self.register_buffer('posterior_log_variance_clipped', posterior_log_variance_clipped)
        self.register_buffer('posterior_mean_coef1', posterior_mean_coef1)
        self.register_buffer('posterior_mean_coef2', posterior_mean_coef2)
        # self.register_buffer('posterior_mean_coef3', posterior_mean_coef3) # only for mu from prior

        self.conditioner = ESMConditioner(config.esm_conditioner, global_config)
        self.x0_pred_net = LatentDiffModel(config, global_config, self.conditioner.embed_dim)
        # self.x0_pred_net = LatentDiffModelSelfCond(config, global_config)
        self.if_norm_latent = if_norm_latent = getattr(self.global_config.latentembedder, 'norm_latent', False)
        
        if if_norm_latent:
            self.latent_scale = 1.0
        else:
            self.latent_scale = self.global_config.latent_scale # 3.2671
        

    def q_sample(self, x0_dict: dict, t, noising_scale=1.0):
        # Calculations for posterior q(x_{t} | x_0, mu)
        xt_dict = {}
        if x0_dict.__contains__('latent_rep'):
            xt_esm = self.degrad_latent(x0_dict['latent_rep'], t, noising_scale=noising_scale)
            xt_dict['latent_rep'] = xt_esm

        return xt_dict

    
    def q_posterior(self, xt_dict, x0_dict, t, noising_scale=1.0):
        # Calculations for posterior q(x_{t-1} | x_t, x_0, mu)
        q_posterior_dict = {}
        t = t[0]
        if x0_dict.__contains__('latent_rep'):
            posterior_mean = self.posterior_mean_coef1[t] * (x0_dict['latent_rep']) + self.posterior_mean_coef2[t] * (xt_dict['latent_rep'])
            model_log_variance = self.posterior_log_variance_clipped[t]

            eps = torch.randn_like(posterior_mean) if t > 0 else torch.zeros_like(posterior_mean)
            x_t_1 = posterior_mean + eps * (0.5 * model_log_variance).exp() * noising_scale
            q_posterior_dict['latent_rep'] = x_t_1

        return q_posterior_dict


    def degrad_latent(self, latent_0, t, noising_scale=1.0):
        device = latent_0.device
        t1 = t[..., None, None]

        noise = torch.randn_like(latent_0) * self.sqrt_one_minus_alphas_cumprod[t1] * noising_scale * self.latent_scale
        degraded_latent = latent_0 * self.sqrt_alphas_cumprod[t1] + noise

        return degraded_latent
    

    def forward(self, batch: dict, return_structure=False, mixed_nn=False):
        device = batch['gt_backbone_pos'].device
        dtype = batch['gt_backbone_pos'].dtype
        batch_size, L = batch['gt_backbone_pos'].shape[:2]
        batch['latent_rep_gt'] = self.x0_pred_net.ldm.x_embedder.wtb(batch['str_code'])
        make_mask(batch['len'], batch_size, L, batch, dtype)

        condition_embed = self.conditioner(batch)
        batch['condition_embed'] = condition_embed
        t = torch.randint(0, self.T, (batch_size,), device=device).long()
        batch['t'] = t
        batch['sqrt_alphas_cumprod'] = self.sqrt_alphas_cumprod[t]
        x0_dict = {
            'latent_rep': batch['latent_rep_gt']
        }
        batch['x0_dict'] = x0_dict

        xt_dict = self.q_sample(batch['x0_dict'], t)
        batch['xt_dict'] = xt_dict
        pred_dict = self.x0_pred_net(batch, return_structure)
        
        losses = {}
        if_weight_loss = getattr(self.global_config, 'weight_loss', False)
        weight_loss_bias = getattr(self.global_config, 'weight_loss_bias', 0.0)
        
        if if_weight_loss:
            weight_loss = batch['sqrt_alphas_cumprod'] + weight_loss_bias
        else:
            weight_loss = torch.ones_like(batch['sqrt_alphas_cumprod'])

        latent_l = latent_loss(pred_dict['pred_latent'], x0_dict['latent_rep'], batch['single_mask'], weight_loss)
        losses.update(latent_l)
        if pred_dict.__contains__('aatype_logits'):
            latent_l = nll_loss(pred_dict['aatype_logits'], batch['aatype'], batch['single_mask'], weight_loss)
            losses.update(latent_l)
        if pred_dict.__contains__('l2_distance'):
            l2_distance = pred_dict['l2_distance']
            l2_distance_mode = getattr(self.global_config, 'l2_distance_mode', 'inverse')
            l2_distance_l = l2_distance_loss(l2_distance, batch['str_code'], batch['single_mask'], l2_distance_mode, weight_loss)
            losses.update(l2_distance_l)
        if pred_dict.__contains__('affine_p'):
            affine_p = pred_dict['affine_p'].float()[None]
            affine_0 = r3.rigids_to_quataffine_m(
                r3.rigids_from_tensor_flat12(batch['gt_backbone_frame'].float())).to_tensor()[..., 0, :]
            fape_losses, fape_dict = fape_loss_multichain(
                affine_p, affine_0, batch['gt_backbone_pos'][..., :3, :].float(), batch['single_mask'], batch['chain_idx'], self.global_config.fape)
            losses.update(fape_losses)

        losses_dict = mask_loss(batch['loss_mask'], losses)
        loss = sum([losses_dict[k].mean() * 
                        self.global_config.loss_weight[k] \
                for k in self.global_config.loss_weight if k in losses_dict.keys()])
        losses_dict['loss'] = loss
        losses_dict = loss_dict_fp32(losses_dict)

        if return_structure:
            assert pred_dict.__contains__('affine_p')
            return losses_dict, fape_dict
        else:
            return losses_dict, None


    @torch.no_grad()
    def sampling(
        self, 
        batch: dict, 
        pdb_prefix: str, 
        step_num: int, 
        init_noising_scale=1.0,
        diff_noising_scale=1.0,
        mapping_nn=False
        ):
        device = batch['aatype'].device
        batch_size, num_res = batch['aatype'].shape[:2]
        latent_dim = self.global_config.in_channels
        latent_rep_nosie = torch.randn((batch_size, num_res, latent_dim), dtype=torch.float32).to(device) * init_noising_scale
        alatent_rep_t = latent_rep_nosie * self.latent_scale
        make_mask(batch['len'], batch_size, num_res, batch, torch.float32)
        condition_embed = self.conditioner(batch)
        if (getattr(self.global_config.loss_weight, "sidechain_embed_loss", 0.0) > 0.0 or (getattr(self.global_config.loss_weight, "sidechain_simi_loss", 0.0) > 0.0) ):
            condition_embed, sc_condtion_rep = condition_embed
        batch['condition_embed'] = condition_embed
        batch['protein_state'] = batch['protein_state'][0]

        xt_dict = {
            'latent_rep': alatent_rep_t
        }
        batch['xt_dict'] = xt_dict

        t_scheme = list(range(self.T-1, -1, -step_num))

        # if t_scheme[-1] != 0:
        #     t_scheme.append(0)

        for t_idx in trange(len(t_scheme)):
            t = t_scheme[t_idx]
            t = torch.LongTensor([t] * batch_size).to(device)
            batch['t'] = t

            x0_dict = self.x0_pred_net(batch, False)
            if not mapping_nn:
                x0_dict['latent_rep'] = x0_dict['pred_latent']
            else:
                x0_dict['latent_rep'] = self.find_nn_latent(x0_dict['pred_latent'])
                        
            x_t_1_dict = self.q_sample(x0_dict, t, noising_scale=diff_noising_scale)
            # x_t_1_dict = self.q_posterior(batch['xt_dict'], x0_dict, t, noising_scale=diff_noising_scale)
            batch['xt_dict'] = x_t_1_dict

        if_norm_latent = getattr(self.global_config.latentembedder, 'norm_latent', False)
        if if_norm_latent:
            x0_dict['latent_rep'] = (x0_dict['latent_rep'] * self.x0_pred_net.ldm.x_embedder.wtb_std[None, None] ) + \
                self.x0_pred_net.ldm.x_embedder.wtb_mean[None, None]

        return x0_dict


    def find_nn_latent(self, pred_latent):
        wtb_weight = self.x0_pred_net.ldm.x_embedder.wtb.weight
        l2_distance = torch.sum((pred_latent[..., None, :] - wtb_weight[None, None])**2, -1)  # B, L, N
        nn_token = torch.argmin(l2_distance, dim=-1)
        nn_latent_rep = F.embedding(nn_token, wtb_weight) # B, L, H

        return nn_latent_rep

    
    @torch.no_grad()
    def sampling_guidance(
        self, 
        batch: dict, 
        pdb_prefix: str, 
        step_num: int, 
        init_noising_scale=1.0,
        diff_noising_scale=1.0,
        mapping_nn=False,
        guide_scale = 1.0,
        guide_scheme='constant',
        guide_fn=None,
        pred_idx=None,
        target_coords=None,
        target_idx=None
        ):
        device = batch['aatype'].device
        batch_size, num_res = batch['aatype'].shape[:2]
        latent_dim = self.global_config.in_channels
        latent_rep_nosie = torch.randn((1, num_res, latent_dim), dtype=torch.float32).to(device) * init_noising_scale
        alatent_rep_t = latent_rep_nosie * self.latent_scale
        make_mask(batch['len'], batch_size, num_res, batch, torch.float32)
        batch['protein_state'] = batch['protein_state'][0]

        xt_dict = {
            'latent_rep': alatent_rep_t
        }
        batch['xt_dict'] = xt_dict

        t_scheme = list(range(self.T-1, -1, -step_num))
        if t_scheme[-1] != 0:
            t_scheme.append(0)

        guide_decay_scheme = guide_scale_scheme(guide_scheme, self.T)

        for t_idx in range(len(t_scheme)):
            t = t_scheme[t_idx]
            t = torch.LongTensor([t] * batch_size).to(device)
            batch['t'] = t

            x0_dict = self.x0_pred_net(batch, False)
            if not mapping_nn:
                x0_dict['latent_rep'] = x0_dict['pred_latent']
            else:
                x0_dict['latent_rep'] = self.find_nn_latent(x0_dict['pred_latent'])
                        
            # x_t_1_dict = self.q_sample(x0_dict, t, noising_scale=diff_noising_scale)
            x_t_1_dict = self.q_posterior(batch['xt_dict'], x0_dict, t, noising_scale=diff_noising_scale)

            x_t_1_guide, traj_rmsd = rmsd_gradient(
                batch['xt_dict']['latent_rep'].clone(), pred_idx, target_coords[..., 1, :], target_idx, batch, guide_fn )
            # x_t_1_guide, traj_fape = fape_gradient(
            #     batch['xt_dict']['latent_rep'].clone(), pred_idx, target_coords, target_idx, batch, guide_fn, self.global_config.fape )
            cur_scale = guide_decay_scheme(t[0]) * guide_scale
            print(t_idx, traj_rmsd, cur_scale.item())
            x_t_1_latent_rep = update_xt_with_grad(x_t_1_dict['latent_rep'], cur_scale, x_t_1_guide)
            x_t_1_dict['latent_rep'] = x_t_1_latent_rep
            batch['xt_dict'] = x_t_1_dict

        return x0_dict


