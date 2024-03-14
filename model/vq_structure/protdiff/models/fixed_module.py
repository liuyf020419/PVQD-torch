import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('/train14/superbrain/yfliu25/ProteinMPNN/vanilla_proteinmpnn')
from protein_mpnn_utils import ProteinMPNN

from .folding_af2.residue_constants import restype_order_with_x
from .protein_utils.add_o_atoms import batch_add_atom_O

# mpnnalphabet = 'ACDEFGHIKLMNPQRSTVWYX'
# alphafold_to_mpnn = {k: mpnnalphabet.index(k) for k in restype_order_with_x.keys()}

checkpoint_path_default = f'/train14/superbrain/yfliu25/ProteinMPNN/vanilla_proteinmpnn/vanilla_model_weights/v_48_020.pt'


class ProteinMPNNModule(nn.Module):
    def __init__(self, checkpoint_path=None, hidden_dim=128, num_layers=3 , backbone_noise=0.00) -> None:
        super().__init__()

        if checkpoint_path is None:
            checkpoint_path = checkpoint_path_default
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        self.fixbbmodel = ProteinMPNN(
            num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim, 
            num_encoder_layers=num_layers, num_decoder_layers=num_layers, 
            augment_eps=backbone_noise, k_neighbors=checkpoint['num_edges'])


    def forward(self, batch, pred_dict, weight=None):
        coords_with_O = pred_dict['coord'][-1]
        device = coords_with_O.device
        batch_size, L, atom_num = coords_with_O.shape[:3]
        assert atom_num == 4
        # coords_with_O = batch_add_atom_O(coords_with_O[..., :3, :])
        if weight is None:
            weight = 1.0
        else:
            if len(weight.shape) == 1:
                weight = weight[:, None]
            assert len(weight.shape) == 2

        # import pdb; pdb.set_trace()
        seq_encoded = batch['mpnn_aatype']

        mask = batch['seq_mask']
        chain_M = torch.ones((batch_size, L)).to(device)
        chain_M_pos = torch.ones((batch_size, L)).to(device)
        randn_1 = torch.randn(chain_M.shape, device=device)
        chain_label = torch.zeros((batch_size, L)).to(device)
        log_probs = self.fixbbmodel(
            coords_with_O, seq_encoded, mask, chain_M*chain_M_pos, 
            batch['single_res_rel'], chain_label, randn_1) # B, L, 20

        nll_loss = F.nll_loss(
            log_probs.contiguous().view(-1,log_probs.size(-1)), 
            seq_encoded.contiguous().view(-1), reduction='none').view(seq_encoded.size())
        # import pdb; pdb.set_trace()
        reduced_nll_loss = torch.sum(nll_loss * mask * weight) / (torch.sum(mask) + 1e-6)

        return reduced_nll_loss

