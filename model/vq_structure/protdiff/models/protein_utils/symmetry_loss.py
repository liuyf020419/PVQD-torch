import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from ..folding_af2 import all_atom
from ..folding_af2 import residue_constants
from ..folding_af2.layers_batch import Linear

from .backbone import backbone_fape_loss, coord_to_frame_affine
from .rigid import quat_to_rot



class PseudoResidueResnetBlock(nn.Module):
    def __init__(self, config):
        """
        Args:
            c_hidden:
                Hidden channel dimension
        """
        super(PseudoResidueResnetBlock, self).__init__()

        self.c_hidden = config.c_hidden

        self.linear_1 = Linear(self.c_hidden, self.c_hidden)
        self.act = nn.GELU()
        self.linear_2 = Linear(self.c_hidden, self.c_hidden)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_0 = x

        x = self.act(x)
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)

        return x + x_0



class PseudoResidueEmbedder(nn.Module):
    def __init__(
        self,
        config: dict,
        **kwargs,
    ):
        """
        Args:
            c_in:
                Input channel dimension
            c_out:
                Output channel dimension
        """
        super(PseudoResidueEmbedder, self).__init__()

        self.d_in = config.d_in
        self.d_out = config.d_out
        self.d_hidden = config.d_hidden
        self.num_blocks = config.num_blocks

        self.sym_group_emb = nn.Embedding(6, self.d_in)
        self.sym_au_num_emb = nn.Embedding(2, self.d_in)
        self.linear_in = Linear(self.d_in, self.d_hidden)
        self.act = nn.GELU()

        self.layers = nn.ModuleList()
        for _ in range(self.num_blocks):
            layer = PseudoResidueResnetBlock(c_hidden=self.d_hidden)
            self.layers.append(layer)

        self.linear_out = Linear(self.d_hidden, self.d_out)

    def forward(self, batch, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                [*, C_in] pseudo residue feature
        Returns:
            [*, C_out] embedding
        """
        sym_au_num_feat = self.sym_au_num_emb(batch['symmetry_group_au_num_encode'])
        sym_group_feat = self.sym_group_emb(batch['symmetry_group_encode'])
        x = sym_au_num_feat + sym_group_feat

        x = x.type(self.linear_in.weight.dtype)

        x = self.linear_in(x)
        x = self.act(x)
        for l in self.layers:
            x = l(x)
        x = self.linear_out(x)

        return x


def merge_symmetry_ops(symmetry_ops_list: list):
    device = symmetry_ops_list[0][0].device
    max_NO_num = max([symmetry_ops[1].shape[0] for symmetry_ops in symmetry_ops_list])
    symmetry_rot_list = []
    symmetry_trans_list = []
    symmetry_mask_list = []
    for symmetry_ops in symmetry_ops_list:
        rot, trans = symmetry_ops
        pad_NO_num = max_NO_num - trans.shape[0]
        paded_rot = torch.cat([
            rot, torch.eye(3).to(device)[None].repeat(pad_NO_num, 1, 1)])
        paded_trans = torch.cat([
            trans, torch.zeros((1, 3)).to(device).repeat(pad_NO_num, 1)])
        paded_mask = F.pad(torch.ones((trans.shape[0], )), (0, pad_NO_num), 'constant', 0)
        symmetry_mask_list.append(paded_mask)

        symmetry_rot_list.append(paded_rot)
        symmetry_trans_list.append(paded_trans)

    symmetry_rot_list = torch.stack(symmetry_rot_list)
    symmetry_trans_list = torch.stack(symmetry_trans_list)
    symmetry_mask_list = torch.stack(symmetry_mask_list) 

    return (symmetry_rot_list, symmetry_trans_list), symmetry_mask_list


def gen_assembly_data_from_AU(
    pred_ori_coord, # B, NR, NA, 3
    native_ori_coord, # B, NR, NA, 3
    symmtry_ops, # rot: B, NO, 3, 3; trans: B, NO, 3
    symmtry_mask, # B, NO,
    residue_index, # B, NR,
    single_mask # B, NR
    ):
    """
    from [B, NR, ...] -> [B, NO * NR, ...]
    """
    batch_size, NO = symmtry_mask
    _, NR, NA = pred_ori_coord.shape[:3]
    au_rots, au_trans = symmtry_ops
    au_tran = au_trans
    au_rot = au_rots

    pred_assembly_coord = (pred_ori_coord[:, None] @ au_rot[..., None, None].float()) + au_tran[..., None, None]
    pred_assembly_coord = (symmtry_mask * pred_assembly_coord).reshape(batch_size, NO * NR, NA, 3)
    native_assembly_coord = (native_ori_coord[:, None] @ au_rot[..., None, None].float()) + au_tran[..., None, None]
    native_assembly_coord = (symmtry_mask * native_assembly_coord).reshape(batch_size, NO * NR, NA, 3)
    assembly_residue_index = expand_residue_index(residue_index, NO)
    assembly_single_mask = (single_mask[:, None].repeat(1, NO, 1) * symmtry_mask[..., None]).reshape(batch_size, -1)

    return {
        'pred_assembly_coord': pred_assembly_coord,
        'native_assembly_coord': native_assembly_coord,
        'assembly_residue_index': assembly_residue_index,
        'assembly_single_mask': assembly_single_mask
    }


def expand_residue_index(residue_index, expand_num, gap_size=20):
    # residue_index.shape (B, NR)
    single_res_rel_end = residue_index[:, -1]
    decollated_single_res_rel_idx = torch.cat([ 
        au_idx * (single_res_rel_end + gap_size) + residue_index 
            for au_idx in range(expand_num) 
        ], -1)

    return decollated_single_res_rel_idx # (B, NO * NR)
    

def get_asym_mask(asym_id):
    """get the mask for each asym_id. [*, NR] -> [*, NC, NR]"""
    # this func presumes that valid asym_id ranges [1, NC] and is dense.
    asym_type = torch.arange(1, torch.amax(asym_id) + 1, device=asym_id.device)  # [NC]
    return (asym_id[..., None, :] == asym_type[:, None]).float()


def symmetry_fape_loss_batch(
    assembly_pred_coord, # B, NR * NO, NA, 3
    assembly_native_coord, # B, NR * NO, NA, 3
    single_mask, # B, NR * NO
    entity_id, # B, NR * NO
    config):

    entity1_res_mask = torch.any(entity_id == 1, dim=-1).float() # B, NR * NO

    pair_au_pred_affine = coord_to_frame_affine(assembly_pred_coord)['affine']   
    pair_au_native_affine = coord_to_frame_affine(assembly_native_coord)['affine']   

    pair_au_pred_quat = pair_au_pred_affine[..., :4]
    pair_au_pred_trans = pair_au_pred_affine[..., 4:]
    pair_au_pred_rot = quat_to_rot(pair_au_pred_quat)

    pair_au_native_quat = pair_au_native_affine[..., :4]
    pair_au_native_trans = pair_au_native_affine[..., 4:]
    pair_au_native_rot = quat_to_rot(pair_au_native_quat)

    fape_loss_pair_mask = single_mask[:, None] * single_mask[:, :, None] # B, NR * NO, NR * NO
    fape_loss_pair_mask = fape_loss_pair_mask * entity1_res_mask[..., None]

    decollated_fape = backbone_fape_loss(
        assembly_pred_coord, pair_au_pred_rot, pair_au_pred_trans,
        assembly_native_coord, pair_au_native_rot, pair_au_native_trans, 
        clamp_dist=config.symmetry_fape.clamp_distance,
        length_scale=config.symmetry_fape.loss_unit_distance,
        mask_2d=fape_loss_pair_mask)

    return decollated_fape

    

def symmetry_clash_loss_batch(
    assembly_pred_coord, # B, NR * NO, NA, 3
    single_mask, # B, NR * NO
    entity_id, # B, NR * NO
    residue_index, # B, NR * NO
    overlap_tolerance_soft,
    overlap_tolerance_hard
    ):

    entity1_res_mask = torch.any(entity_id == 1, dim=-1).float() # B, NR * NO
    batch_size, NRall, NA = assembly_pred_coord.shape[:3]

    sym_loss_pair_mask = (single_mask[:, None] * single_mask[:, :, None]) * (entity1_res_mask[:, None] * entity1_res_mask[:, :, None])
    sym_loss_pair_mask = sym_loss_pair_mask[..., None, None]

    atom14_pred_positions= assembly_pred_coord.float()
    # batchsize, L = atom14_pred_positions.shape[:2]
    # num_atoms = np.prod(list(atom14_pred_positions.shape[:-1])).astype(np.float32)
    # seq_mask: (B, N)
    # pred_atom_mask: (B, N, natoms)
    pred_atom_mask = single_mask[..., None].repeat(1, 1, NA)

    # Compute the Van der Waals radius for every atom
    # (the first letter of the atom name is the element type).
    # Shape: (N, 14).
    atomtype_radius = torch.FloatTensor([
        residue_constants.van_der_waals_radius[name[0]]
        for name in residue_constants.atom_types
    ]).to(atom14_pred_positions.device)
    # num_res = L
    # (B, N, natoms)
    atom14_atom_radius = pred_atom_mask * atomtype_radius[None, None].repeat(batch_size, NRall, 1)[..., :NA]

    # Compute the between residue clash loss.
    between_residue_clashes = all_atom.between_residue_clash_loss_batch(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=pred_atom_mask,
        atom14_atom_radius=atom14_atom_radius,
        residue_index=residue_index,
        overlap_tolerance_soft=overlap_tolerance_soft,
        overlap_tolerance_hard=overlap_tolerance_hard,
        natoms=NA,
        pair_mask=sym_loss_pair_mask)

    between_residue_clash_loss = between_residue_clashes['per_atom_loss_sum'],  # (N, 14)

    violation_loss = between_residue_clash_loss # /(1e-6 + num_atoms)
    
    return violation_loss



def symmetry_center_mass_loss_batch(
    assembly_pred_coord, # B, NR * NO, NA, 3
    assembly_native_coord, # B, NR * NO, NA, 3
    single_mask, # B, NR * NO
    asym_id, # B, NR * NO
    entity_id, # B, NR * NO
    eps: float = 1e-10):
    """
    NR: AU_len, NO: Ops_num, NC: Chain_num
    e.g. ER: 12; NO: 2; NC: 6
    """

    asym_mask = get_asym_mask(asym_id) * single_mask[..., None, :] # B, NC, NR * NO
    entity_asym = get_asym_mask(asym_id) * entity_id[..., None, :] # B, NC， NR * NO
    entity1_chain_mask = torch.any(entity_asym == 1, dim=-1).float() # B, NC
    asym_exists = torch.any(asym_mask, dim=-1).float()  # [B, NC]
    pred_atom_positions = assembly_pred_coord[..., 1, :].float()  # [B, NR * NO, 3]
    true_atom_positions = assembly_native_coord[..., 1, :].float()  # [B, NR * NO, 3]
    # batch_size, NC = asym_mask.shape[:2]
    # device = asym_mask.device

    def get_asym_centres(pos):
        pos = pos[..., None, :, :] * asym_mask[..., :, :, None]  # [B, NC, NR * NO, 3]
        return torch.sum(pos, dim=-2) / (torch.sum(asym_mask, dim=-1)[..., None] + eps)
    
    pred_centres = get_asym_centres(pred_atom_positions)  # [B, NC, 3]
    true_centres = get_asym_centres(true_atom_positions)  # [B, NC, 3]

    def get_dist(p1: torch.Tensor, p2: torch.Tensor):
        return torch.sqrt(
            (p1[..., :, None, :] - p2[..., None, :, :]).square().sum(-1) + eps
        )

    pred_centres2 = pred_centres
    true_centres2 = true_centres
    pred_dists = get_dist(pred_centres, pred_centres2)  # [B, NC, NC]
    true_dists = get_dist(true_centres, true_centres2)  # [B, NC, NC]
    losses = (pred_dists - true_dists + 4).clamp(max=0).square() * 0.0025
    loss_mask = asym_exists[..., :, None] * asym_exists[..., None, :]  # [B, NC, NC]
    loss_mask = loss_mask * entity1_chain_mask[..., None]

    center_mass_loss = torch.sum(loss_mask * losses, dim=(-1, -2)) / (eps + torch.sum(loss_mask, dim=(-1, -2)))

    return center_mass_loss



def center_mass_loss_batch(
    assembly_pred_coord, # B, NR * NO, NA, 3
    assembly_native_coord, # B, NR * NO, NA, 3
    single_mask, # B, NR * NO
    asym_id, # B, NR * NO
    eps: float = 1e-10):
    """
    NR: AU_len, NO: Ops_num, NC: Chain_num
    e.g. ER: 12; NO: 2; NC: 6
    """
    # import pdb; pdb.set_trace()
    monomer_mask = torch.any(asym_id > 0, 1)
    asym_mask = get_asym_mask(asym_id + 1) * single_mask[..., None, :] # B, NC, NR * NO
    asym_exists = torch.any(asym_mask, dim=-1).float()  # [B, NC]
    pred_atom_positions = assembly_pred_coord[..., 1, :].float()  # [B, NR * NO, 3]
    true_atom_positions = assembly_native_coord[..., 1, :].float()  # [B, NR * NO, 3]
    # batch_size, NC = asym_mask.shape[:2]
    # device = asym_mask.device

    def get_asym_centres(pos):
        pos = pos[..., None, :, :] * asym_mask[..., :, :, None]  # [B, NC, NR * NO, 3]
        return torch.sum(pos, dim=-2) / (torch.sum(asym_mask, dim=-1)[..., None] + eps)
    
    pred_centres = get_asym_centres(pred_atom_positions)  # [B, NC, 3]
    true_centres = get_asym_centres(true_atom_positions)  # [B, NC, 3]

    def get_dist(p1: torch.Tensor, p2: torch.Tensor):
        return torch.sqrt(
            (p1[..., :, None, :] - p2[..., None, :, :]).square().sum(-1) + eps
        )

    pred_centres2 = pred_centres
    true_centres2 = true_centres
    pred_dists = get_dist(pred_centres, pred_centres2)  # [B, NC, NC]
    true_dists = get_dist(true_centres, true_centres2)  # [B, NC, NC]
    losses = (pred_dists - true_dists + 4).clamp(max=0).square() * 0.0025
    loss_mask = asym_exists[..., :, None] * asym_exists[..., None, :]  # [B, NC, NC]
    # loss_mask = loss_mask * entity1_chain_mask[..., None]
    center_mass_loss = torch.sum((loss_mask * losses)[monomer_mask]) / ( eps + torch.sum(loss_mask[monomer_mask]) )

    return center_mass_loss



def symmetry_fape_loss(pred_coord, native_coord, single_mask, symmetry_ops, config):
    """
    only calculate between au fape loss, exculding O1 -> O1
    """
    au_rots, au_trans = symmetry_ops
    au_num = au_rots.shape[0]
    pred_ori_coord = pred_coord
    native_ori_coord = native_coord
    device = pred_coord.device
    batch_size, au_length = pred_coord.shape[:2]

    pair_au_single_mask = single_mask.repeat(1, 2).to(device)

    pair_au_clash_mask = torch.ones(batch_size, 2 * au_length, 2 * au_length).to(device)
    pair_au_clash_mask[:, :au_length, :au_length] = torch.zeros(batch_size, au_length, au_length).to(device)
    pair_au_clash_mask[:, au_length:, au_length:] = torch.zeros(batch_size, au_length, au_length).to(device)
    pair_au_fape_mask = pair_au_clash_mask[..., None, None]

    between_au_fape_loss = []

    for au_idx in range(au_num-1):
        au_idx = au_idx + 1
        au_tran = au_trans[au_idx]
        au_rot = au_rots[au_idx]

        pred_au_sym_coord = (pred_ori_coord[..., 4:] @ au_rot.float()) + au_tran
        native_au_sym_coord = (native_ori_coord[..., 4:] @ au_rot.float()) + au_tran

        pair_au_pred_coord = torch.cat([
            pred_ori_coord, pred_au_sym_coord], -3)
        pair_au_native_coord = torch.cat([
            native_ori_coord, native_au_sym_coord], -3)

        pair_au_pred_affine = coord_to_frame_affine(pair_au_pred_coord)['affine']   
        pair_au_native_affine = coord_to_frame_affine(pair_au_native_coord)['affine']   

        pair_au_pred_quat = pair_au_pred_affine[..., :4]
        pair_au_pred_trans = pair_au_pred_affine[..., 4:]
        pair_au_pred_rot = quat_to_rot(pair_au_pred_quat)

        pair_au_native_quat = pair_au_native_affine[..., :4]
        pair_au_native_trans = pair_au_native_affine[..., 4:]
        pair_au_native_rot = quat_to_rot(pair_au_native_quat)


        pair_au_fape = backbone_fape_loss(
            pair_au_pred_coord, pair_au_pred_rot, pair_au_pred_trans,
            pair_au_native_coord, pair_au_native_rot, pair_au_native_trans, 
            pair_au_single_mask,
            clamp_dist=config.symmetry_fape.clamp_distance,
            length_scale=config.symmetry_fape.loss_unit_distance,
            mask_2d=pair_au_fape_mask) # N^2 mean has been calculate inside func

        between_au_fape_loss.append(pair_au_fape)

    between_au_fape_loss = torch.cat(between_au_fape_loss).sum()

    return (1 / (au_num - 1)) * between_au_fape_loss



def symmetry_clash_loss(batch, atom14_pred_positions, symmetry_ops, config):
    au_rots, au_trans = symmetry_ops
    au_num = au_rots.shape[0]
    origin_au_atom14_pred_positions = atom14_pred_positions
    batch_size, au_length = atom14_pred_positions.shape[:2]
    device = atom14_pred_positions.device
    
    au_single_res_rel_end = batch['single_res_rel'][:, -1]
    pair_au_residue_index = torch.cat([ 
        au_idx * (au_single_res_rel_end + config.pair_au_gap_size) + batch['single_res_rel'] 
            for au_idx in range(2) ], -1)[0]

    pair_au_single_mask = batch['seq_mask'].repeat(1, 2).to(device)

    pair_au_clash_mask = torch.ones(batch_size, 2 * au_length, 2 * au_length).to(device)
    pair_au_clash_mask[:, :au_length, :au_length] = torch.zeros(batch_size, au_length, au_length).to(device)
    pair_au_clash_mask[:, au_length:, au_length:] = torch.zeros(batch_size, au_length, au_length).to(device)
    pair_au_atom_clash_mask = pair_au_clash_mask[..., None, None]
    
    between_au_violation_loss = []

    for au_idx in range(au_num-1):
        au_idx = au_idx + 1
        au_tran = au_trans[au_idx]
        au_rot = au_rots[au_idx]
        aym_au_atom14_pred_positions = (origin_au_atom14_pred_positions @ au_rot.float()) + au_tran

        pair_au_atom14_pred_positions = torch.cat([
            origin_au_atom14_pred_positions, aym_au_atom14_pred_positions], -3)
        
        pair_au_violation_loss = pair_au_clash_loss(
            pair_au_single_mask, pair_au_residue_index, 
            pair_au_atom_clash_mask, pair_au_atom14_pred_positions, config)

        between_au_violation_loss.append(pair_au_violation_loss)

    between_au_violation_loss = torch.cat(between_au_violation_loss).sum() / (torch.sum(pair_au_atom_clash_mask) + 1e-6) 

    return (1 / (au_num - 1)) * between_au_violation_loss




def pair_au_clash_loss(single_mask, residue_index, pair_au_atom_clash_mask, atom14_pred_positions, config):
    atom14_pred_positions= atom14_pred_positions.float()
    batchsize, L, atoms_num = atom14_pred_positions.shape[:3]
    # num_atoms = np.prod(list(atom14_pred_positions.shape[:-1])).astype(np.float32)
    # seq_mask: (B, N)
    # pred_atom_mask: (B, N, natoms)
    pred_atom_mask = single_mask[..., None].repeat(1, 1, atoms_num)

    # Compute the Van der Waals radius for every atom
    # (the first letter of the atom name is the element type).
    # Shape: (N, 14).
    atomtype_radius = torch.FloatTensor([
        residue_constants.van_der_waals_radius[name[0]]
        for name in residue_constants.atom_types
    ]).to(atom14_pred_positions.device)
    num_res = L
    # (B, N, natoms)
    atom14_atom_radius = pred_atom_mask * atomtype_radius[None, None].repeat(batchsize, num_res, 1)[..., :atoms_num]

    # Compute the between residue clash loss.
    between_residue_clashes = all_atom.between_residue_clash_loss_batch(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=pred_atom_mask,
        atom14_atom_radius=atom14_atom_radius,
        residue_index=residue_index,
        overlap_tolerance_soft=config.clash_overlap_tolerance,
        overlap_tolerance_hard=config.clash_overlap_tolerance,
        natoms=atoms_num,
        pair_mask=pair_au_atom_clash_mask)

    between_residue_clash_loss = between_residue_clashes['per_atom_loss_sum'],  # (N, 14)

    violation_loss = between_residue_clash_loss # /(1e-6 + num_atoms)
    
    return violation_loss


def symmetry_centroid_loss(pred_affine, native_affine, symmetry_ops, centroid_config):
    centoid_tolerance_factor = centroid_config.centoid_tolerance_factor
    centoid_loss_scaler = centroid_config.centoid_loss_scaler

    native_centroid_coord = []
    pred_centroid_coord = []
    au_rots, au_trans = symmetry_ops
    au_num = au_rots.shape[0]

    for au_idx in range(au_num):
        au_tran = au_trans[au_idx]
        au_rot = au_rots[au_idx]

        pred_au_sym_coord = (pred_affine[..., 4:] @ au_rot.float()) + au_tran
        native_au_sym_coord = (native_affine[..., 4:] @ au_rot.float()) + au_tran

        pred_centroid_coord.append(pred_au_sym_coord.mean(-2))
        native_centroid_coord.append(native_au_sym_coord.mean(-2))

    native_centroid_coord = torch.stack(native_centroid_coord, 1)
    pred_centroid_coord = torch.stack(pred_centroid_coord, 1)

    native_centroid_dist_map = native_centroid_coord[:, :, None] - native_centroid_coord[:, None]
    pred_centroid_dist_map = pred_centroid_coord[:, :, None] - pred_centroid_coord[:, None]

    centroid_dist_loss_ = torch.abs(pred_centroid_dist_map - native_centroid_dist_map) - centoid_tolerance_factor
    centroid_dist_loss = (1/centoid_loss_scaler) *  torch.max(centroid_dist_loss_, 0)

    return (1 / (au_num-1) * au_num ** 2) * centroid_dist_loss


