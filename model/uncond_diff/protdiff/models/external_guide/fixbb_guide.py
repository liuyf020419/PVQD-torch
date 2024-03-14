import os

import numpy as np

import torch

from fixbb_score_guide_utils import mpnn_gradient, merge_guide_grad

import sys
sys.path.append('/train14/superbrain/yfliu25/ProteinMPNN/vanilla_proteinmpnn')
from protein_mpnn_utils import ProteinMPNN


class FixBBGuidance(object):
    def __init__(self, fixbb_name, checkpoint_path=None, device=None) -> None:
        super().__init__()
        if device is not None:
            self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        else:
            self.device = device
        self.fixbb_name = fixbb_name

        if fixbb_name == 'abacusr':
            self.fixbbmodel = None
        elif fixbb_name == 'proteinmpnn':
            hidden_dim = 128
            num_layers = 3 
            backbone_noise = 0.00
            if checkpoint_path is None:
                checkpoint_path = f'/train14/superbrain/yfliu25/ProteinMPNN/vanilla_proteinmpnn/vanilla_model_weights/v_48_020.pt'
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu")) 
            self.fixbbmodel = ProteinMPNN(
                num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim, 
                num_encoder_layers=num_layers, num_decoder_layers=num_layers, 
                augment_eps=backbone_noise, k_neighbors=checkpoint['num_edges'])
            self.fixbbmodel.load_state_dict(checkpoint['model_state_dict'])
        elif fixbb_name == 'gvp':
            self.fixbbmodel = None
        else:
            raise ValueError(f'{fixbb_name} unknown')

        self.fixbbmodel.eval()
        self.fixbbmodel.to(device)


    def compute_gradient(self, X, S, residue_idx, chain_idx, selected, guide_dict):
        guide_mode = guide_dict['fixbb_mode']
        assert guide_mode in ['selected_S', 'selected_entropy', 'selected_logp']
        if self.fixbb_name == 'proteinmpnn':
            gradient = mpnn_gradient(X, S, residue_idx, chain_idx, self.fixbbmodel, guide_mode, selected)
        elif self.fixbb_name == 'abacusr':
            pass
        elif self.fixbb_name == 'gvp':
            pass

        return gradient


    def update_coordinates(self, X, S, residue_idx, chain_idx, guide_mode: str, selected=None, guide_scale=1.0):
        gradient = self.compute_gradient(self, X, S, residue_idx, chain_idx, guide_mode, selected)
        new_coords = merge_guide_grad(X, guide_scale, gradient)
        return new_coords
