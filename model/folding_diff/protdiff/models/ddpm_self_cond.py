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
from .nn_utils import make_mask, mask_loss, latent_loss, fape_loss_multichain, l2_distance_loss, get_coords_dict_from_affine
# from .latent_diff_model import LatentDiffModel
from .latent_diff_model_self_cond import LatentDiffModel as LatentDiffModelSelfCond


logger = logging.getLogger(__name__)


def loss_dict_fp32(dict:dict):
    fp32_dict = {}
    for k, v in dict.items():
        fp32_dict[k] = v.float()

    return fp32_dict


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
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        posterior_variance = torch.stack([posterior_variance, torch.FloatTensor([1e-20] * self.T)])
        posterior_log_variance_clipped = posterior_variance.max(dim=0).values.log()
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        posterior_mean_coef1 = betas * alphas_cumprod_prev.sqrt() / (1 - alphas_cumprod)
        posterior_mean_coef2 = (1 - alphas_cumprod_prev) * alphas.sqrt() / (1 - alphas_cumprod)
        # posterior_mean_coef3 = (1 - (betas * alphas_cumprod_prev.sqrt() + alphas.sqrt() * (1 - alphas_cumprod_prev))/ (1 - alphas_cumprod)) # only for mu from prior
        self.register_buffer('posterior_log_variance_clipped', posterior_log_variance_clipped)
        self.register_buffer('posterior_mean_coef1', posterior_mean_coef1)
        self.register_buffer('posterior_mean_coef2', posterior_mean_coef2)
        # self.register_buffer('posterior_mean_coef3', posterior_mean_coef3) # only for mu from prior

        self.x0_pred_net = LatentDiffModelSelfCond(config, global_config)
        if_norm_latent = getattr(self.global_config.latentembedder, 'norm_latent', False)
        
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


    def degrad_latent(self, latent_0, t, noising_scale=1.0):
        device = latent_0.device
        t1 = t[..., None, None]

        noise = torch.randn_like(latent_0) * self.sqrt_one_minus_alphas_cumprod[t1] * noising_scale * self.latent_scale
        degraded_latent = latent_0 * self.sqrt_alphas_cumprod[t1] + noise

        return degraded_latent
    

    def forward(self, batch: dict, return_structure=False):
        device = batch['gt_backbone_pos'].device
        dtype = batch['gt_backbone_pos'].dtype
        batch_size, L = batch['gt_backbone_pos'].shape[:2]
        batch['latent_rep_gt'] = self.x0_pred_net.ldm.x_embedder.wtb(batch['str_code'])
        make_mask(batch['len'], batch_size, L, batch, dtype)

        t = torch.randint(0, self.T, (batch_size,), device=device).long()
        batch['t'] = t
        batch['sqrt_alphas_cumprod'] = self.sqrt_alphas_cumprod[t]
        x0_dict = {
            'latent_rep': batch['latent_rep_gt']
        }
        batch['x0_dict'] = x0_dict

        if self.global_config.self_condition:
            tprev_mask = torch.where(t+1 >= self.T, 0, 1)
            tprev_mask = (tprev_mask * (torch.rand_like(batch['t'].float()) < 0.5).long()).bool()
            xprevt_dict = self.q_sample(batch['x0_dict'], torch.where(t+1 >= self.T, t, t+1))
            batch['xt_dict'] = xprevt_dict
            xprev0_rep = self.x0_pred_net(batch, return_structure, None)['pred_latent'].detach()
            xprev0_rep = torch.where(tprev_mask[:, None, None], xprev0_rep, torch.zeros_like(xprev0_rep)).detach()
        else:
            xprev0_rep = None

        xt_dict = self.q_sample(batch['x0_dict'], t)
        batch['xt_dict'] = xt_dict
        pred_dict = self.x0_pred_net(batch, return_structure, xprev0_rep)
        
        losses = {}
        if_weight_loss = getattr(self.global_config, 'weight_loss', False)
        if if_weight_loss:
            weight_loss = batch['sqrt_alphas_cumprod']
        else:
            weight_loss = torch.ones_like(batch['sqrt_alphas_cumprod'])

        latent_l = latent_loss(pred_dict['pred_latent'], x0_dict['latent_rep'], batch['single_mask'], weight_loss)
        losses.update(latent_l)
        if pred_dict.__contains__('l2_distance'):
            l2_distance = pred_dict['l2_distance']
            l2_distance_mode = getattr(self.global_config, 'l2_distance_mode', 'inverse')
            l2_distance_l = l2_distance_loss(l2_distance, batch['str_code'], batch['single_mask'], l2_distance_mode, weight_loss)
            losses.update(l2_distance_l)
        if pred_dict.__contains__('affine_p'):
            affine_p = pred_dict['affine_p'].float()[None]
            affine_0 = r3.rigids_to_quataffine_m(r3.rigids_from_tensor_flat12(batch['gt_backbone_frame'].float())).to_tensor()[..., 0, :]
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

        for t_idx in trange(len(t_scheme)):
            t = t_scheme[t_idx]
            t = torch.LongTensor([t] * batch_size).to(device)
            batch['t'] = t

            x0_dict = self.x0_pred_net(batch, False)
            x0_dict['latent_rep'] = x0_dict['pred_latent']

            # if (t[0] == 3):
            #     return x0_dict
                # coord = get_coords_dict_from_affine(x0_dict['affine_p'][None])['coord']
                # for batch_idx in range(batch_size):
                #     traj_coord_0 = add_atom_O(coord[0, batch_idx].detach().cpu().numpy()[..., :3, :])
                #     write_multichain_from_atoms([traj_coord_0.reshape(-1, 3)], 
                #         f'{pdb_prefix}_diff_{t[0].item()}_scale_{diff_noising_scale}_batch_{batch_idx}.pdb', natom=4)
                        
            x_t_1_dict = self.q_sample(x0_dict, t, noising_scale=diff_noising_scale)
            batch['xt_dict'] = x_t_1_dict

        if_norm_latent = getattr(self.global_config.latentembedder, 'norm_latent', False)
        if if_norm_latent:
            x0_dict['latent_rep'] = (x0_dict['latent_rep'] * self.x0_pred_net.ldm.x_embedder.wtb_std[None, None] ) + \
                self.x0_pred_net.ldm.x_embedder.wtb_mean[None, None]

        return x0_dict


    @torch.no_grad()
    def sampling_bak(
        self, 
        batch: dict, 
        pdb_prefix: str, 
        step_num: int, 
        mu_dict: dict=None, 
        return_traj=False, 
        init_noising_scale=1.0, 
        ddpm_fix=False, 
        rigid_fix=False, 
        diff_noising_scale=1.0,
        post_sampling=False
        ):

        device = batch['aatype'].device
        batch_size, num_res = batch['aatype'].shape[:2]
        fape_condition = None
        rigid_fix_align_freq = 7

        if mu_dict is None:
            affine_tensor_nosie = torch.randn((1, num_res, 6), dtype=torch.float32).to(device) * init_noising_scale
            affine_tensor_t = affine_tensor_nosie * self.affine_tensor_scale.to(device)
            affine_t = affine6_to_affine7(affine_tensor_t)

            esm_t = torch.randn(
                (1, num_res, self.global_config.esm_num), dtype=torch.float32).to(device)
        else:
            affine_tensor_noise = torch.randn((1, num_res, 6), dtype=torch.float32).to(device) * init_noising_scale
            affine_tensor_t = affine7_to_affine6(mu_dict['affine']) + affine_tensor_noise * self.affine_tensor_scale.to(device)
            affine_t = affine6_to_affine7(affine_tensor_t)

            if not ddpm_fix:
                affine_t = torch.where(batch['condition'][..., None]==1, batch['traj_affine'][..., 0, :], affine_t)

            esm_noise = torch.randn(
                (1, num_res, self.global_config.esm_num), dtype=torch.float32).to(device) * init_noising_scale
            esm_t = mu_dict['esm'] + esm_noise

        if ddpm_fix:
            fix_condition = batch['condition']
            batch['condition'] = torch.zeros_like(batch['condition'])

        xt_dict = {
            'affine': affine_t,
            'esm': esm_t
        }
        batch['xt_dict'] = xt_dict
        
        if not batch.__contains__('gt_pos'):
            batch['gt_pos'] = mu_dict['coord']
            affine_0 = mu_dict['affine']
        else:
            affine_0 = r3.rigids_to_quataffine_m(
                r3.rigids_from_tensor_flat12(batch['gt_backbone_frame'])
                ).to_tensor()[..., 0, :]

        t_scheme = list(range(self.T-1, -1, -step_num))
        # t_scheme = ((np.exp(np.arange(100) * 0.02) - np.exp(np.arange(100) * 0.02)[0])/np.exp(np.arange(100) * 0.02)[-1] * 400).astype(np.int32)[::-1]
        # t_scheme = ((np.exp(np.arange(50) * 0.05) - np.exp(np.arange(50) * 0.05)[0])/np.exp(np.arange(50) * 0.05)[-1] * 400).astype(np.int32)[::-1]
        esm_pred_list = []
        if t_scheme[-1] != 0:
            t_scheme.append(0)
        for t_idx, t in enumerate(t_scheme):
            t = torch.LongTensor([t] * batch_size).to(device)
            batch['t'] = t
            # import pdb; pdb.set_trace()
            # write_multichain_from_atoms([add_atom_O(self.affine_to_coord(batch['xt_dict']['affine']).detach().cpu().numpy()[..., :3, :]).reshape(-1, 3)], f'/train14/superbrain/yfliu25/structure_refine/debug_PriorDiff_evo2_fixaffine_fixfape_condition/trash/{t[0].item()}_xt.pdb', natom=4)
            x0_dict = self.x0_pred_net(batch)
            x0_dict = {k: v[-1] if k == 'traj' else v for k, v in x0_dict.items()}
            # x0_dict['affine'] = x0_dict['traj']
            if not ddpm_fix:
                fape_condition = batch['condition']
            else:
                fape_condition = fix_condition

            # generate traj and logger
            affine_p = x0_dict['traj']
            losses, pred_x0_dict = self.fape_loss(
                affine_p[None], affine_0, 
                batch['gt_pos'][..., :3, :], batch['seq_mask'], 
                fape_condition)
            fape_loss = losses['fape_loss'].item()
            clamp_fape = losses['clamp_fape_loss'].item()

            if batch.__contains__('norm_esm_single'):
                esm_loss = self.esm_loss(batch, x0_dict['esm'], batch['norm_esm_single'])['esm_single_pred_loss'].item()
                logger.info(f'step: {t[0].item()}/{self.T}; fape loss: {round(fape_loss, 3)}; clamp fape: {round(clamp_fape, 3)}; esm loss: {round(esm_loss, 3)}')
            else:
                logger.info(f'step: {t[0].item()}/{self.T}; fape loss: {round(fape_loss, 3)}; clamp fape: {round(clamp_fape, 3)}')
            esm_pred_list.append(x0_dict['esm'].detach().cpu().numpy())
            
            # import pdb; pdb.set_trace()
            if return_traj:
                if t[0] == 3:
                    # gt_aatype_af2idx = batch['aatype'][0].detach().cpu().numpy().tolist()
                    # gt_aatype_str = ''.join([af2_index_to_aatype[aa] for aa in gt_aatype_af2idx])
                    # fasta_dict = {'native_seq': gt_aatype_str}
                        
                    for batch_idx in range(batch_size):
                        traj_coord_0 = add_atom_O(pred_x0_dict['coord'][0, batch_idx].detach().cpu().numpy()[..., :3, :])
                        write_multichain_from_atoms([traj_coord_0.reshape(-1, 3)], 
                            f'{pdb_prefix}_diff_{t[0].item()}_scale_{diff_noising_scale}_batch_{batch_idx}.pdb', natom=4)
                        # if x0_dict.__contains__('aatype'):
                        #     pred_aatype_logits = torch.argmax(x0_dict['aatype'][batch_idx], -1).reshape((-1, )).detach().cpu().numpy()
                        #     pred_aatype_str = ''.join([af2_index_to_aatype[aa] for aa in pred_aatype_logits])
                        # else:
                        #     pred_aatype_str = predict_aatype(predictor, x0_dict['esm'].detach().cpu()[batch_idx])
                        # fasta_dict.update({f'predicted_{batch_idx}': pred_aatype_str})

                    # fasta_writer(fasta_f=f'{pdb_prefix}_diff_{t[0].item()}_scale_{diff_noising_scale}.fasta', fasta_dict=fasta_dict)
                    # logger.info('esm1b prediction saved')
                    # import pdb; pdb.set_trace()
                    # out_dict = {}
                    # out_dict['atom3_coords'] = pred_x0_dict['coord'][0].detach().cpu().numpy()
                    # out_dict['raw_pdbabslID'] = batch['reprint_resabsID'][0][None].detach().cpu().numpy()
                    # np.save(f'{pdb_prefix}_diff_{t[0].item()}_scale_{diff_noising_scale}_monomer_coords.npy', out_dict)

            if ddpm_fix:
                if fix_condition is not None:
                    if rigid_fix:
                        x0_dict_affine = x0_dict['traj']
                        if t_idx % rigid_fix_align_freq == 0:
                            for batch_idx in range(batch_size):
                                # mobile part of coord_gt to part of coord_pred
                                rotransed_gt_pos = kabschalign.align(
                                    batch['gt_pos'][batch_idx][fix_condition[0] == 1][:, :3], 
                                    pred_x0_dict['coord'][0, batch_idx][fix_condition[0] == 1],
                                    cycles=1, verbose=False)
                                rotransed_affine_0 = coord_to_frame_affine(rotransed_gt_pos)['affine'][0]
                        # replace part of coord_pred with part of coord_gt
                        x0_dict_affine[batch_idx][fix_condition[0] == 1] = rotransed_affine_0
                        x0_dict['affine'] = x0_dict_affine
                    else:
                        x0_dict['affine'] = torch.where(fix_condition[..., None] == 1, affine_0, x0_dict['traj'])
                else:
                    x0_dict['affine'] = x0_dict['traj']
            else:
                x0_dict['affine'] = x0_dict['traj']

            if post_sampling:
                if not ddpm_fix:
                    self.q_posterior(batch['xt_dict'], x0_dict, t, mu_dict, noising_scale=diff_noising_scale)
                else:
                    self.q_posterior(batch['xt_dict'], x0_dict, t, mu_dict, batch['condition'], affine_0[..., None, :], noising_scale=diff_noising_scale)
            else:
                if not ddpm_fix:
                    x_t_1_dict = self.q_sample(x0_dict, t, mu_dict, batch['condition'], affine_0[..., None, :], noising_scale=diff_noising_scale)
                else:
                    x_t_1_dict = self.q_sample(x0_dict, t, mu_dict, noising_scale=diff_noising_scale)
            batch['xt_dict'] = x_t_1_dict

        x0_dict = batch['xt_dict']

        return x0_dict
    

    