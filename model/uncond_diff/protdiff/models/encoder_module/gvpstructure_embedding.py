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

        if self.conitnious_res_num == 1:
            self.embed_gvp_input_features = nn.Linear(6 + 9, embed_dim)
        else:
            self.embed_gvp_input_features = nn.Linear((6 + 9) * self.conitnious_res_num, embed_dim)
        self.embed_confidence = nn.Linear(16, embed_dim)
        self.embed_dihedrals = DihedralFeatures(embed_dim, self.conitnious_res_num)
        self.aatype_embedding = nn.Embedding(22, embed_dim)
        if self.conitnious_res_num > 1:
            self.aatype_embedding_continuous = nn.Linear(embed_dim * self.conitnious_res_num, embed_dim)

        gvp_args = self.config.gvp
        self.gvp_encoder = GVPEncoder(gvp_args)
        gvp_out_dim = (gvp_args.node_hidden_dim_scalar + (3 *
                gvp_args.node_hidden_dim_vector)) * self.conitnious_res_num
        self.embed_gvp_output = nn.Linear(gvp_out_dim, embed_dim)


    def forward(self, batch):
        dtype = batch['traj_pos'].dtype
        coord_dict = {
            'coord': batch['traj_pos'].float(),
            'backbone_frame': batch['traj_backbone_frame'].float()
        }
        data_dict = self.make_data_dict(coord_dict, batch['single_mask'], batch['single_res_rel'])

        coords = data_dict['coord'][..., :3, :]
        batchsize, res_num = coords.shape[:2]
        device = coords.device

        padding_mask = ~batch['single_mask'].bool()
        confidence = data_dict['confidence']

        # if self.global_config.aatype_embedding:
        #     aatype = batch['aatype']
        # else:
        #     aatype = torch.zeros_like(data_dict['aatype']).long()
        if (self.global_config.aatype_embedding and self.global_config.aatype_embedding_in_encoder):
            if not self.training:
                aatype_drop_p = 0.0
                aatype_replace_p = 0.0
                aatype_maintain_p = 0.0
            else:
                aatype_drop_p = self.global_config.aatype_drop_p
                aatype_replace_p = self.global_config.aatype_replace_p
                aatype_maintain_p = self.global_config.aatype_maintain_p

            aatype_mask = (torch.rand_like(batch['single_mask']) > aatype_drop_p).float() * batch['single_mask']
            batch['aatype_mask'] = (1 - aatype_mask) * batch['single_mask']
            input_aatype = batch['aatype']
            aatype_replace_unmask = 1 - ( (torch.rand_like(batch['single_mask']) * (1 - aatype_mask)) > aatype_replace_p).float() * batch['single_mask']
            aatype_maintain_replace_unmask = 1 - ( (torch.rand_like(batch['single_mask']) * (1 - aatype_mask) * (1 - aatype_replace_unmask) ) > aatype_maintain_p).float() * batch['single_mask']

            aatype_replace_only = (aatype_replace_unmask - aatype_mask) * batch['single_mask']
            aatype_maintain_only = (aatype_maintain_replace_unmask - aatype_replace_unmask) * batch['single_mask']

            aatype = (input_aatype * aatype_mask + (1 - aatype_mask) * 21).long()
            aatype = torch.where(aatype_replace_only.bool(), torch.randint_like(aatype, 0, 20), aatype)
            aatype = torch.where(aatype_maintain_only.bool(), input_aatype, aatype)

        else:
            aatype = torch.zeros_like(batch['single_mask']).long()
            batch['aatype_mask'] = None

        # R = data_dict['rot']
        R = get_rotation_frames(coords)
        # import pdb; pdb.set_trace()
        res_idx = data_dict['res_idx']
        res_idx = res_idx * ~padding_mask + self.global_config.pad_num*padding_mask
        # import pdb; pdb.set_trace()

        coord_mask = torch.all(torch.all(torch.isfinite(coords), dim=-1), dim=-1)
        coords = nan_to_num(coords)

        # GVP encoder out
        gvp_out_scalars, gvp_out_vectors = self.gvp_encoder(coords,
                coord_mask, res_idx, padding_mask, confidence)
        # import pdb; pdb.set_trace()
        if self.conitnious_res_num == 1:
            gvp_out_features = torch.cat([
                gvp_out_scalars,
                rotate(gvp_out_vectors, R.transpose(-2, -1)).flatten(-2, -1),
            ], dim=-1)
            # import pdb; pdb.set_trace()
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
        components["diherals"] = self.embed_dihedrals(coords)

        components["gvp_out"] = self.embed_gvp_output(gvp_out_features)
        # components["confidence"] = self.embed_confidence(rbf(confidence, 0., 1.))
        scalar_features, vector_features = GVPInputFeaturizer.get_node_features(
            coords, coord_mask, with_coord_mask=False)

        if self.conitnious_res_num > 1:
            components["tokens"] = self.aatype_embedding_continuous(
                (self.aatype_embedding(aatype) * self.embed_scale).reshape(
                    batchsize, len(low_resolution_select_idx), -1))
        else:
            components["tokens"] = self.aatype_embedding(aatype) * self.embed_scale

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

        components["gvp_input_features"] = self.embed_gvp_input_features(features)
        # import pdb; pdb.set_trace()
        embed = sum(components.values())

        return embed.to(dtype)


    def make_data_dict(self, coord_dict: dict, seq_mask_traj, res_idx):
        batchsize, L = coord_dict['coord'].shape[:2]
        coord = coord_dict['coord'][..., :3, :]
        if not coord_dict.__contains__('rot'):
            new_shape = list(coord_dict['backbone_frame'].shape[:-2]) + [3, 3]
            rot = coord_dict['backbone_frame'][..., 0, :9].reshape(new_shape)
            rot = rot.reshape(batchsize, L, 3, 3)
        else:
            rot = coord_dict['rot']
        pseudo_aatype = torch.zeros(batchsize, L).long().to(coord.device)
        data_dict = {'coord': coord, 'encoder_padding_mask': seq_mask_traj.bool(), 
                    'confidence': torch.ones(batchsize, L).to(coord.device), 'rot': rot, 
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

    def forward(self, coords, coord_mask, res_idx, padding_mask, confidence):
        node_embeddings, edge_embeddings, edge_index = self.embed_graph(
                coords, coord_mask, res_idx, padding_mask, confidence)
        for i, layer in enumerate(self.encoder_layers):
            node_embeddings, edge_embeddings = layer(node_embeddings,
                    edge_index, edge_embeddings)

        node_embeddings = unflatten_graph(node_embeddings, coords.shape[0])
        return node_embeddings
