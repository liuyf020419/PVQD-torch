import logging
import numpy as np

import torch
import torch.nn as nn

from .ddpm import DDPM
from .nn_utils import mask_loss

from .protein_utils.write_pdb import write_multichain_from_atoms
from .protein_utils.add_o_atoms import add_atom_O

logger = logging.getLogger(__name__)


class PriorDDPM(nn.Module):
    def __init__(self, config, global_config, data_config) -> None:
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.data_config = data_config

        self.diff_module = DDPM(self.config, self.global_config)


    def forward(self, batch):
        losses_dict = {}

        diff_loss_dict = self.diff_module(batch)
        diff_loss_dict = {'diff_'+k: v for k, v in diff_loss_dict.items()}
        losses_dict.update(diff_loss_dict)

        losses_dict = mask_loss(batch['loss_mask'], losses_dict)
        loss = sum([losses_dict[k].mean() * 
                        self.global_config.loss_weight[k] \
                for k in self.global_config.loss_weight if k in losses_dict.keys()])
        losses_dict['loss'] = loss

        return pred_dict, losses_dict


    @torch.no_grad()
    def sampling(
        self, 
        batch, 
        pdb_prefix, 
        diff_step: int, 
        noising_mode_idx: int, 
        condition=None, 
        return_traj=False, 
        symmetry: str=None,
        ddpm_fix=False, 
        rigid_fix=False, 
        diff_noising_scale=1.0, 
        no_prior_sampling=False, 
        post_sampling=False
        ):
        batchsize, L, N, _ = batch['traj_pos'].shape
        make_mask(batch['len'], batchsize, L, batch)

        # mu_dict = self.prior_module.sampling(batch, pdb_prefix, noising_mode_idx, condition)
        # x0_dict = self.diff_module.sampling(batch, pdb_prefix, diff_step, mu_dict, return_traj, symmetry=symmetry)
        # import pdb; pdb.set_trace()
        if no_prior_sampling:
            mu_dict = None
        else:
            mu_dict = self.prior_module.sampling(batch, pdb_prefix, noising_mode_idx, condition)

        x0_dict = self.diff_module.sampling(
            batch, pdb_prefix, diff_step, mu_dict, return_traj, ddpm_fix=ddpm_fix, rigid_fix=rigid_fix, \
                diff_noising_scale=diff_noising_scale, post_sampling=post_sampling)
        # generate x0 feature pdb and aatype
        if return_traj is False:
            pred_coord4 = add_atom_O(x0_dict['coord'].detach().cpu().numpy()[0])
            write_multichain_from_atoms([pred_coord4.reshape(-1, 3)], f'{pdb_prefix}_last_{diff_step}.pdb')






