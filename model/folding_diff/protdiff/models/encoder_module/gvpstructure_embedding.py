import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

from ..esm.modules import SinusoidalPositionalEmbedding
from ..esm.inverse_folding.transformer_layer import TransformerEncoderLayer
from ..esm.inverse_folding.gvp_transformer_encoder import nan_to_num, rbf, norm, normalize, rotate, get_rotation_frames
from ..esm.inverse_folding.features import GVPGraphEmbedding, DihedralFeatures, GVPInputFeaturizer
from ..esm.inverse_folding.gvp_modules import GVPConvLayer, LayerNorm 
from ..esm.inverse_folding.gvp_utils import unflatten_graph
from torch.utils.checkpoint import checkpoint
# from ..esm.inverse_folding.util import get_rotation_frames


class GVPStructureEmbedding(nn.Module):
    def __init__(self, config, global_config, conitnious_res_num=1) -> None:
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.conitnious_res_num = conitnious_res_num
        self.pad_res_num = self.conitnious_res_num//2

        self.embed_dim = embed_dim = self.config.encoder_embed_dim
        self.embed_scale = math.sqrt(embed_dim)

        gvp_args = self.config.gvp
        self.gvp_args = gvp_args

        self.sidechain_rep = getattr(gvp_args, 'sidechain_rep', False)

        if (not self.sidechain_rep):
            if getattr(gvp_args, 'all_atom', False):
                in_embed_gvp_input_features = 6 + 39 * 3
            else:
                in_embed_gvp_input_features = 6 + 3 * 3
        else:
            in_embed_gvp_input_features = 64 + 6 * 3
        if self.conitnious_res_num == 1:
            self.embed_gvp_input_features = nn.Linear(in_embed_gvp_input_features, embed_dim)
        else:
            self.embed_gvp_input_features = nn.Linear(in_embed_gvp_input_features * self.conitnious_res_num, embed_dim)
        self.embed_confidence = nn.Linear(16, embed_dim)
        self.embed_dihedrals = DihedralFeatures(embed_dim, self.conitnious_res_num)
        self.aatype_embedding = nn.Embedding(22, embed_dim)
        if self.conitnious_res_num > 1:
            self.aatype_embedding_continuous = nn.Linear(embed_dim * self.conitnious_res_num, embed_dim)
        
        self.gvp_encoder = GVPEncoder(gvp_args)
        gvp_out_dim = (gvp_args.node_hidden_dim_scalar + (3 *
                gvp_args.node_hidden_dim_vector)) * self.conitnious_res_num
        self.embed_gvp_output = nn.Linear(gvp_out_dim, embed_dim)


    def forward(self, batch, condition_mask, mixed_nn=False):
        dtype = batch['gt_backbone_pos'].dtype
        device = batch['gt_backbone_pos'].device
        padding_mask = ~batch['single_mask'].bool()
        gvp_mask = ~condition_mask.bool()

        if self.sidechain_rep:
            sidechain_function_pos = batch['sidechain_function_pos']
            sidechain_function_coords_mask = batch['sidechain_function_coords_mask']
            coord_dict = {
                    'coord': sidechain_function_pos.float() * condition_mask[..., None, None]
                }
            data_dict = self.make_data_dict(coord_dict, condition_mask, batch['single_res_rel'])
            # import pdb; pdb.set_trace()
            coord_mask = torch.all(sidechain_function_coords_mask, -1).long()
            coord_mask = (coord_mask * condition_mask).bool()

            backbone_coords = sidechain_function_pos.float() * coord_mask[..., None, None]
            batchsize, res_num = backbone_coords.shape[:2]
            confidence = data_dict['confidence']
            
            R = get_rotation_frames(backbone_coords)
            res_idx = data_dict['res_idx']
            res_idx = res_idx * ~padding_mask + self.global_config.pad_num*padding_mask

            coords = nan_to_num(backbone_coords)

        else:
            if getattr(self.gvp_args, "all_atom", False):
                coord_dict = {
                    'coord': batch['gt_backbone_pos_atom37'].float() * condition_mask[..., None, None],
                    'backbone_frame': batch['gt_backbone_frame'].float() * condition_mask[..., None, None]
                }
            else:
                coord_dict = {
                    'coord': batch['gt_backbone_pos'].float() * condition_mask[..., None, None],
                    'backbone_frame': batch['gt_backbone_frame'].float() * condition_mask[..., None, None]
                }
            data_dict = self.make_data_dict(coord_dict, condition_mask, batch['single_res_rel'])

            backbone_coords = batch['gt_backbone_pos'][..., :3, :].float() * condition_mask[..., None, None]
            batchsize, res_num = backbone_coords.shape[:2]
            confidence = data_dict['confidence']
            # import pdb; pdb.set_trace()
            R = get_rotation_frames(backbone_coords)
            res_idx = data_dict['res_idx']
            res_idx = res_idx * ~padding_mask + self.global_config.pad_num*padding_mask
            
            # import pdb; pdb.set_trace()
            if getattr(self.gvp_args, "all_atom", False):
                atom37_coords = batch['gt_backbone_pos_atom37'].float() * condition_mask[..., None, None]
                coord_mask = torch.all(torch.all(torch.isfinite(atom37_coords), dim=-1), dim=-1).long()
                coord_mask = (coord_mask * condition_mask).bool()
                coords = nan_to_num(atom37_coords)
            else:
                coord_mask = torch.all(torch.all(torch.isfinite(backbone_coords), dim=-1), dim=-1).long()
                coord_mask = (coord_mask * condition_mask).bool()
                coords = nan_to_num(backbone_coords)

        # GVP encoder out
        gvp_aatype = None
        if (getattr(self.gvp_args, "aatype_embed", False) or self.sidechain_rep):
            gvp_aatype = (batch['aatype'] * condition_mask).long()
            aatype = (batch['aatype'] * condition_mask).long()
        else:
            aatype = torch.zeros_like(condition_mask).long()
        batch['aatype_mask'] = None
        all_atom = getattr(self.gvp_args, "all_atom", False)
        all_atom_mask = None
        if all_atom:
            all_atom_mask = batch['atom37_mask']
        # import pdb; pdb.set_trace()
        gvp_out_scalars, gvp_out_vectors = self.gvp_encoder(coords,
                coord_mask, res_idx, gvp_mask, confidence, gvp_aatype, all_atom=all_atom, all_atom_mask=all_atom_mask, mixed_nn=mixed_nn)
        if self.conitnious_res_num == 1:
            gvp_out_features = torch.cat([
                gvp_out_scalars,
                rotate(gvp_out_vectors, R.transpose(-2, -1)).flatten(-2, -1),
            ], dim=-1)
        else:
            # # # BxLxD
            low_resolution_select_idx = torch.arange(0, res_num, self.conitnious_res_num).long().to(device)
            expanded_gvp_out_scalars = gvp_out_scalars.reshape(batchsize, len(low_resolution_select_idx), -1)

            # BxLxLxNx3
            rotated_vector_mat = torch.sum(gvp_out_vectors[:, None, ..., None] * R[:, :, None, ..., None, :, :], -2)
            # BxLxCxNx3 -> LxBxCxNx3
            expanded_gvp_out_vectors = torch.stack(
                [rotated_vector_mat[:, first_v_idx, first_v_idx + torch.arange(self.conitnious_res_num).to(device)] \
                    for first_v_idx in low_resolution_select_idx],1)
            expanded_gvp_out_vectors = expanded_gvp_out_vectors.reshape(batchsize, len(low_resolution_select_idx), -1)

            gvp_out_features = torch.cat([
                expanded_gvp_out_scalars,expanded_gvp_out_vectors], dim=-1)

        components = dict()

        # raw feature
        if (not self.sidechain_rep):
            components["diherals"] = self.embed_dihedrals(backbone_coords)

        components["gvp_out"] = self.embed_gvp_output(gvp_out_features)
        if self.sidechain_rep:
            scalar_features, vector_features = self.gvp_encoder.embed_graph._get_sidechain_node_feature(gvp_aatype, coords)
        else:
            scalar_features, vector_features = GVPInputFeaturizer.get_node_features(
                coords, coord_mask, with_coord_mask=False, all_atom=all_atom, all_atom_mask=all_atom_mask)

        if self.conitnious_res_num > 1:
            components["tokens"] = self.aatype_embedding_continuous(
                (self.aatype_embedding(aatype) * self.embed_scale).reshape(
                    batchsize, len(low_resolution_select_idx), -1))
        else:
            components["tokens"] = self.aatype_embedding(aatype) * self.embed_scale * condition_mask[..., None]

        if self.conitnious_res_num == 1:
            features = torch.cat([
                scalar_features,
                rotate(vector_features, R.transpose(-2, -1)).flatten(-2, -1),
            ], dim=-1)
        else:
            # BxLxD
            low_resolution_select_idx = torch.arange(0, res_num, self.conitnious_res_num).long().to(device)
            expanded_scalar_features = scalar_features.reshape(batchsize, len(low_resolution_select_idx), -1)

            # BxLxLxNx3
            rotated_vector_features_mat = torch.sum(vector_features[:, None, ..., None] * R[:, :, None, ..., None, :, :], -2)
            # BxLxCxNx3 -> LxBxCxNx3
            expanded_vector_features = torch.stack(
                [rotated_vector_features_mat[:, first_v_idx, first_v_idx + torch.arange(self.conitnious_res_num).to(device)] \
                    for first_v_idx in low_resolution_select_idx],1)
            expanded_vector_features = expanded_vector_features.reshape(batchsize, len(low_resolution_select_idx), -1)
            # import pdb; pdb.set_trace()
            features = torch.cat([expanded_scalar_features,expanded_vector_features], dim=-1)

        features = features * condition_mask[..., None]
        components["gvp_input_features"] = self.embed_gvp_input_features(features)

        embed = sum(components.values())

        return embed.to(dtype)


    def make_data_dict(self, coord_dict: dict, seq_mask_traj, res_idx):
        batchsize, L = coord_dict['coord'].shape[:2]
        coord = coord_dict['coord'][..., :3, :]
        pseudo_aatype = torch.zeros(batchsize, L).long().to(coord.device)
        data_dict = {'coord': coord, 'encoder_padding_mask': seq_mask_traj.bool(), 
                    'confidence': torch.ones(batchsize, L).to(coord.device),
                    'res_idx': res_idx,
                    'aatype': pseudo_aatype}
        
        return data_dict



class GVPEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_graph = GVPGraphEmbedding(args)

        node_hidden_dim = (args.node_hidden_dim_scalar,
                args.node_hidden_dim_vector)
        edge_hidden_dim = (args.edge_hidden_dim_scalar,
                args.edge_hidden_dim_vector)
        
        conv_activations = (F.relu, torch.sigmoid)
        self.encoder_layers = nn.ModuleList(
                GVPConvLayer(
                    node_hidden_dim,
                    edge_hidden_dim,
                    drop_rate=args.dropout,
                    vector_gate=True,
                    attention_heads=0,
                    n_message=3,
                    conv_activations=conv_activations,
                    n_edge_gvps=0,
                    eps=1e-4,
                    layernorm=True,
                ) 
            for i in range(args.num_encoder_layers)
        )

    def forward(self, coords, coord_mask, res_idx, padding_mask, confidence, aatype=None, all_atom=False, all_atom_mask=None, mixed_nn=False):
        node_embeddings, edge_embeddings, edge_index = self.embed_graph(
                coords, coord_mask, res_idx, padding_mask, confidence, aatype, all_atom, all_atom_mask, mixed_nn)
        
        for i, layer in enumerate(self.encoder_layers):
            if getattr(self.args, "gradient_checkpointing", False) and self.training:
                node_embeddings, edge_embeddings = checkpoint(
                    layer.forward_checkpoint, 
                    node_embeddings[0], node_embeddings[1], edge_index, edge_embeddings[0], edge_embeddings[1]
                )
            else:
                node_embeddings, edge_embeddings = layer(
                    node_embeddings, edge_index, edge_embeddings
                )

        node_embeddings = unflatten_graph(node_embeddings, coords.shape[0])
        return node_embeddings
