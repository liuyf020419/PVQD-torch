from typing import Union
import math

import torch
import torch.nn as nn

from .esm.encoder_self_cond import ESM2Encoder, DiTEncoder
from .nn_utils import TransformerPositionEncoding

import sys
sys.path.append('/raw7/superbrain/yfliu25/VQstructure/vqgvp_rvq_multichain/protdiff')
from models.vqstructure import VQStructure


class LatentDiffModel(nn.Module):
    def __init__(self, config, global_config) -> None:
        super().__init__()
        self.config = config
        self.global_config = global_config
        encoder_mode = global_config.encoder_mode
        encoder_mode in ['ESM2Encoder', 'DiTEncoder']

        in_channels = self.global_config.in_channels

        if (encoder_mode == 'ESM2Encoder'):
            self.ldm = ESM2Encoder(config.ESM2Encoder, global_config, in_channels * 2, in_channels)
        elif (encoder_mode == 'DiTEncoder'):
            self.ldm = DiTEncoder(config.DiTEncoder, global_config, in_channels * 2, in_channels)
        else:
            raise ValueError()

        self.single_res_embedding = TransformerPositionEncoding(
            global_config.max_seq_len, self.ldm.config.embed_dim)
        self.single_chain_embedding = nn.Embedding(
            global_config.max_chain_len, self.ldm.config.embed_dim)
        self.single_entity_embedding = nn.Embedding(
            global_config.max_entity_len, self.ldm.config.embed_dim)

        if self.global_config.loss_weight.fape_loss > 0.0:
            self.vqdecoder = VQStructure(
                self.config.vqstucture_model_config, 
                self.config.vqstucture_global_config
                )


    def forward(self, batch, return_structure=False, xprev0=None):
        t = batch['t']
        input_hidden = batch['xt_dict']['latent_rep']
        y = batch['protein_state']
        single_mask = batch['single_mask']
        dtype = input_hidden.dtype

        padding_mask = ~single_mask.bool()
        single_idx = batch['single_res_rel']
        chain_idx = batch['chain_idx']
        entity_idx = batch['entity_idx']
        single_idx = single_idx * ~padding_mask + self.global_config.pad_num*padding_mask
        chain_idx = (chain_idx * ~padding_mask).long() + self.global_config.pad_chain_num*padding_mask
        entity_idx = (entity_idx * ~padding_mask).long() + self.global_config.pad_entity_num*padding_mask
        
        single_condition = self.single_res_embedding(single_idx, index_select=True).to(dtype) + \
            self.single_chain_embedding(chain_idx).to(dtype) + \
                self.single_entity_embedding(entity_idx).to(dtype)

        if xprev0 is None:
            input_hidden = torch.cat([input_hidden, torch.zeros_like(input_hidden)], -1)
        else:
            input_hidden = torch.cat([input_hidden, xprev0], -1)

        pred_latent = self.ldm(None, t, y, single_mask, input_hidden, single_condition)

        pred_dict = {}
        pred_dict['pred_latent'] = pred_latent

        codebook_reps = self.ldm.x_embedder.wtb.weight # N, D
        if self.training:
            assert (codebook_reps.requires_grad == False)
        l2_distance = torch.sum((pred_latent[..., None, :] - codebook_reps[None, None])**2, -1)  # B, L, N
        pred_dict['l2_distance'] = l2_distance

        if ( (self.global_config.loss_weight.fape_loss > 0.0) or return_structure ):
            min_codebook_latent, min_pred_indices, _ = \
                self.vqdecoder.codebook.compute_each_codebook(
                    self.vqdecoder.codebook.codebook_layer[0], pred_latent)
            min_codebook_input = self.vqdecoder.codebook.post_quant(min_codebook_latent)
            reps = self.vqdecoder.decode(
                min_codebook_input, batch['single_mask'], \
                    batch['single_res_rel'], batch['chain_idx'], batch['entity_idx'],\
                    batch['pair_res_idx'], batch['pair_chain_idx'], batch['pair_same_entity'])
            affine_p, single_rep, pair_rep_act = reps
            pred_dict['affine_p'] = affine_p

        return pred_dict


    def decode_structure_from_code(self, batch, indices):
        codebook_mapping = self.vqdecoder.codebook.get_feature_from_indices(indices)

        reps = self.vqdecoder.decode(
            codebook_mapping, batch['single_mask'], \
                batch['single_res_rel'], batch['chain_idx'], batch['entity_idx'],\
                batch['pair_res_idx'], batch['pair_chain_idx'], batch['pair_same_entity'])
        affine_p, single_rep, pair_rep_act = reps
        
        return affine_p
        

