# Copyright (c) Facebook, Inc. and its affiliates.
#
# Portions of this file were adapted from the open source code for the following
# two papers:
#
#   Ingraham, J., Garg, V., Barzilay, R., & Jaakkola, T. (2019). Generative
#   models for graph-based protein design. Advances in Neural Information
#   Processing Systems, 32.
#
#   Jing, B., Eismann, S., Suriana, P., Townshend, R. J. L., & Dror, R. (2020).
#   Learning from Protein Structure with Geometric Vector Perceptrons. In
#   International Conference on Learning Representations.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gvp_utils import flatten_graph
from .gvp_modules import GVP, LayerNorm
# from .util import normalize, norm, nan_to_num, rbf


class GVPInputFeaturizer(nn.Module):

    @staticmethod
    def get_node_features(coords, coord_mask, with_coord_mask=True):
        # scalar features
        node_scalar_features = GVPInputFeaturizer._dihedrals(coords)
        # node_scalar_features = torch.cat(
        #     [torch.sin(quat_angles), torch.cos(quat_angles)], dim=-1
        # )
        if with_coord_mask:
            node_scalar_features = torch.cat([
                node_scalar_features,
                coord_mask.float().unsqueeze(-1)
            ], dim=-1) 
        # vector features
        X_ca = coords[:, :, 1]
        orientations = GVPInputFeaturizer._orientations(X_ca)
        sidechains = GVPInputFeaturizer._sidechains(coords)
        node_vector_features = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
        return node_scalar_features, node_vector_features

    @staticmethod
    def _orientations(X):
        forward = normalize(X[:, 1:] - X[:, :-1])
        backward = normalize(X[:, :-1] - X[:, 1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)
    
    @staticmethod
    def _sidechains(X):
        n, origin, c = X[:, :, 0], X[:, :, 1], X[:, :, 2]
        c, n = normalize(c - origin), normalize(n - origin)
        bisector = normalize(c + n)
        perp = normalize(torch.cross(c, n, dim=-1))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec 

    @staticmethod
    def _dihedrals(X, eps=1e-7):
        X = torch.flatten(X[:, :, :3], 1, 2)
        bsz = X.shape[0]
        dX = X[:, 1:] - X[:, :-1]
        U = normalize(dX, dim=-1)
        u_2 = U[:, :-2]
        u_1 = U[:, 1:-1]
        u_0 = U[:, 2:]
    
        # Backbone normals
        n_2 = normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
        n_1 = normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)
    
        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)
    
        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2]) 
        D = torch.reshape(D, [bsz, -1, 3])
        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], -1)
        return D_features

    @staticmethod
    def _positional_embeddings(edge_index, 
                               res_idx=None,
                               num_embeddings=None,
                               num_positional_embeddings=16,
                               clamp_dist=None,
                               period_range=[2, 1000]):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or num_positional_embeddings
        if res_idx is not None:
            edge_start = torch.gather(res_idx, -1, edge_index[0])
            edge_end = torch.gather(res_idx, -1, edge_index[1])
        else:
            edge_start = edge_index[0]
            edge_end = edge_index[1]
        # d = edge_index[0] - edge_index[1]
        d = edge_start - edge_end
        if clamp_dist is not None:
            clamp_dist = np.abs(clamp_dist)
            d = torch.clamp(d, -clamp_dist, clamp_dist)
     
        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32,
                device=edge_index.device)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    @staticmethod
    def _dist(X, coord_mask, res_idx, padding_mask, top_k_neighbors, eps=1e-8, order=3):
        """ Pairwise euclidean distances """
        bsz, maxlen = X.size(0), X.size(1)
        coord_mask_2D = torch.unsqueeze(coord_mask,1) * torch.unsqueeze(coord_mask,2)
        residue_mask = ~padding_mask
        residue_mask_2D = torch.unsqueeze(residue_mask,1) * torch.unsqueeze(residue_mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = coord_mask_2D * norm(dX, dim=-1)

        cov_dist = ((res_idx[..., None] - res_idx[..., None, :]).abs() <= order).float()
        D_cov = D * (1-cov_dist)
    
        # sorting preference: first those with coords, then among the residues that
        # exist but are masked use distance in sequence as tie breaker, and then the
        # residues that came from padding are last
        seqpos = torch.arange(maxlen, device=X.device)
        Dseq = torch.abs(seqpos.unsqueeze(1) - seqpos.unsqueeze(0)).repeat(bsz, 1, 1)
        D_adjust = nan_to_num(D_cov) + (~coord_mask_2D) * (1e8 + Dseq*1e6) + (
            ~residue_mask_2D) * (1e10)
    
        if top_k_neighbors == -1:
            D_neighbors = D_adjust
            E_idx = seqpos.repeat(
                    *D_neighbors.shape[:-1], 1)
        else:
            # Identify k nearest neighbors (including self)
            k = min(top_k_neighbors, X.size(1))
            D_neighbors, E_idx = torch.topk(D_adjust, k, dim=-1, largest=False)
            D_neighbors = torch.gather(D, -1, E_idx)
    
        coord_mask_neighbors = (D_neighbors < 5e7)
        residue_mask_neighbors = (D_neighbors < 5e9)
        return D_neighbors, E_idx, coord_mask_neighbors, residue_mask_neighbors


class Normalize(nn.Module):
    def __init__(self, features, epsilon=1e-6):
        super(Normalize, self).__init__()
        self.gain = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x, dim=-1):
        mu = x.mean(dim, keepdim=True)
        sigma = torch.sqrt(x.var(dim, keepdim=True) + self.epsilon)
        gain = self.gain
        bias = self.bias
        # Reshape
        if dim != -1:
            shape = [1] * len(mu.size())
            shape[dim] = self.gain.size()[0]
            gain = gain.view(shape)
            bias = bias.view(shape)
        return gain * (x - mu) / (sigma + self.epsilon) + bias


class DihedralFeatures(nn.Module):
    def __init__(self, node_embed_dim, node_angle_len=1):
        """ Embed dihedral angle features. """
        super(DihedralFeatures, self).__init__()
        self.node_angle_len = node_angle_len
        # self.node_pad_num = self.node_angle_len//2
        # 3 dihedral angles; sin and cos of each angle
        node_in = 6 * node_angle_len
        # Normalization and embedding
        self.node_embedding = nn.Linear(node_in,  node_embed_dim, bias=True)
        self.norm_nodes = Normalize(node_embed_dim)

    def forward(self, X, torsion=None):
        """ Featurize coordinates as an attributed graph """
        if torsion is not None:
            V = torch.cat(
                    [torch.sin(torsion), torch.cos(torsion)], dim=-1
                )
        else:
            V = self._dihedrals(X)
            
        V = self.node_embedding(V)
        V = self.norm_nodes(V)
        return V

    # @staticmethod
    def _dihedrals(self, X, eps=1e-7):
        batchsize, res_num = X.shape[:2]
        device = X.device
        # First 3 coordinates are N, CA, C
        X = X[:,:,:3,:].reshape(X.shape[0], 3*X.shape[1], 3)

        # Shifted slices of unit vectors
        dX = X[:,1:,:] - X[:,:-1,:]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:,:-2,:]
        u_1 = U[:,1:-1,:]
        u_0 = U[:,2:,:]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)

        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, (1,2), 'constant', 0)
        D = D.view((D.size(0), int(D.size(1)/3), 3))
        # phi, psi, omega = torch.unbind(D,-1)

        if self.node_angle_len >1:
            select_idx = torch.arange(0, res_num, self.node_angle_len).long().to(device)
            expanded_D = D.reshape(batchsize, len(select_idx), -1)

            D_features = torch.cat((torch.cos(expanded_D), torch.sin(expanded_D)), -1)
        else:
            D_features = torch.cat([torch.cos(D), torch.sin(D)], -1)
        
        return D_features


class GVPGraphEmbedding(GVPInputFeaturizer):

    def __init__(self, args):
        super().__init__()
        self.top_k_neighbors = args.top_k_neighbors
        # my args
        self.edge_cross_vec = args.edge_cross_vec
        self.clamp_rel_pos = args.clamp_rel_pos
        self.num_positional_embeddings = 16
        self.remove_edges_without_coords = True
        node_input_dim = (7, 3)
        if not self.edge_cross_vec:
            edge_input_dim = (34, 1)
        else:
            edge_input_dim = (34, 2)
        node_hidden_dim = (args.node_hidden_dim_scalar,
                args.node_hidden_dim_vector)
        edge_hidden_dim = (args.edge_hidden_dim_scalar,
                args.edge_hidden_dim_vector)
        self.embed_node = nn.Sequential(
            GVP(node_input_dim, node_hidden_dim, activations=(None, None)),
            LayerNorm(node_hidden_dim, eps=1e-4)
        )
        self.embed_edge = nn.Sequential(
            GVP(edge_input_dim, edge_hidden_dim, activations=(None, None)),
            LayerNorm(edge_hidden_dim, eps=1e-4)
        )
        self.embed_confidence = nn.Linear(16, args.node_hidden_dim_scalar)

    def forward(self, coords, coord_mask, res_idx, padding_mask, confidence):
        with torch.no_grad():
            node_features = self.get_node_features(coords, coord_mask)
            edge_features, edge_index = self.get_edge_features(
                coords, coord_mask, res_idx, padding_mask, self.edge_cross_vec)
        node_embeddings_scalar, node_embeddings_vector = self.embed_node(node_features)
        edge_embeddings = self.embed_edge(edge_features)

        rbf_rep = rbf(confidence, 0., 1.)
        # import pdb; pdb.set_trace()
        node_embeddings = (
            node_embeddings_scalar + self.embed_confidence(rbf_rep),
            node_embeddings_vector
        )

        node_embeddings, edge_embeddings, edge_index = flatten_graph(
            node_embeddings, edge_embeddings, edge_index)
        return node_embeddings, edge_embeddings, edge_index

    def get_edge_features(self, coords, coord_mask, res_idx, padding_mask, edge_cross_vec=False):
        X_ca = coords[:, :, 1]
        # Get distances to the top k neighbors
        E_dist, E_idx, E_coord_mask, E_residue_mask = GVPInputFeaturizer._dist(
                X_ca, coord_mask, res_idx, padding_mask, self.top_k_neighbors)
        # Flatten the graph to be batch size 1 for torch_geometric package 
        dest = E_idx
        B, L, k = E_idx.shape[:3]
        src = torch.arange(L, device=E_idx.device).view([1, L, 1]).expand(B, L, k)
        # After flattening, [2, B, E]
        edge_index = torch.stack([src, dest], dim=0).flatten(2, 3)
        # After flattening, [B, E]
        E_dist = E_dist.flatten(1, 2)
        E_coord_mask = E_coord_mask.flatten(1, 2).unsqueeze(-1)
        E_residue_mask = E_residue_mask.flatten(1, 2)
        # Calculate relative positional embeddings and distance RBF 
        pos_embeddings = GVPInputFeaturizer._positional_embeddings(
            edge_index,
            res_idx,
            num_positional_embeddings=self.num_positional_embeddings,
            clamp_dist=self.clamp_rel_pos
        )
        D_rbf = rbf(E_dist, 0., 20.)
        # Calculate relative orientation 
        X_src = X_ca.unsqueeze(2).expand(-1, -1, k, -1).flatten(1, 2)
        X_dest = torch.gather(
            X_ca,
            1,
            edge_index[1, :, :].unsqueeze(-1).expand([B, L*k, 3])
        )
        coord_mask_src = coord_mask.unsqueeze(2).expand(-1, -1, k).flatten(1, 2)
        coord_mask_dest = torch.gather(
            coord_mask,
            1,
            edge_index[1, :, :].expand([B, L*k])
        )
        E_vectors = X_src - X_dest
        # For the ones without coordinates, substitute in the average vector
        E_vector_mean = torch.sum(E_vectors * E_coord_mask, dim=1,
                keepdims=True) / torch.sum(E_coord_mask, dim=1, keepdims=True)
        E_vectors = E_vectors * E_coord_mask + E_vector_mean * ~(E_coord_mask)
        # Normalize and remove nans 
        edge_s = torch.cat([D_rbf, pos_embeddings], dim=-1)
        edge_v = normalize(E_vectors).unsqueeze(-2)

        if edge_cross_vec:
            sidechains =  GVPInputFeaturizer._sidechains(coords)
            sc_src = sidechains.unsqueeze(2).expand(-1, -1, k, -1).flatten(1, 2)
            sc_dest = torch.gather(
                sidechains,
                1,
                edge_index[1, :, :].unsqueeze(-1).expand([B, L*k, 3])
            )
            sc_cross_vectors = torch.cross(sc_src, sc_dest)
            sc_cross_vectors_mean = torch.sum(sc_cross_vectors * E_coord_mask, dim=1,
                keepdims=True) / torch.sum(E_coord_mask, dim=1, keepdims=True)
            sc_cross_vectors = sc_cross_vectors * E_coord_mask + sc_cross_vectors_mean * ~(E_coord_mask)
            edge_sc_v = normalize(sc_cross_vectors).unsqueeze(-2)
            edge_v = torch.cat([edge_v, edge_sc_v], -2)

        edge_s, edge_v = map(nan_to_num, (edge_s, edge_v))
        # Also add indications of whether the coordinates are present 
        edge_s = torch.cat([
            edge_s,
            (~coord_mask_src).float().unsqueeze(-1),
            (~coord_mask_dest).float().unsqueeze(-1),
        ], dim=-1)
        edge_index[:, ~E_residue_mask] = -1
        if self.remove_edges_without_coords:
            edge_index[:, ~E_coord_mask.squeeze(-1)] = -1
        return (edge_s, edge_v), edge_index.transpose(0, 1) 



def nan_to_num(ts, val=0.0):
    """
    Replaces nans in tensor with a fixed value.    
    """
    val = torch.tensor(val, dtype=ts.dtype, device=ts.device)
    return torch.where(~torch.isfinite(ts), val, ts)


def rbf(values, v_min, v_max, n_bins=16):
    """
    Returns RBF encodings in a new dimension at the end.
    """
    rbf_centers = torch.linspace(v_min, v_max, n_bins, device=values.device)
    rbf_centers = rbf_centers.view([1] * len(values.shape) + [-1])
    rbf_std = (v_max - v_min) / n_bins
    v_expand = torch.unsqueeze(values, -1)
    z = (values.unsqueeze(-1) - rbf_centers) / rbf_std
    return torch.exp(-z ** 2)


def norm(tensor, dim, eps=1e-8, keepdim=False):
    """
    Returns L2 norm along a dimension.
    """
    return torch.sqrt(
            torch.sum(torch.square(tensor), dim=dim, keepdim=keepdim) + eps)


def normalize(tensor, dim=-1):
    """
    Normalizes a tensor along a dimension after removing nans.
    """
    return nan_to_num(
        torch.div(tensor, norm(tensor, dim=dim, keepdim=True))
    )

