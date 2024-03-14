import torch
import numpy as np

from alphafold.data.utils import templates, mmcif_parsing, data_transforms
from alphafold.common import residue_constants
from alphafold import all_atom
from alphafold import quat_affine


MAX_TEMPLATE_HITS = 20
residue_atom_renaming_swaps_idx = {
    'ASP': {'OD1': 6, 'OD2': 7},
    'GLU': {'OE1': 7, 'OE2': 8},
    'PHE': {'CD1': 6, 'CD2': 7, 'CE1': 8, 'CE2': 9},
    'TYR': {'CD1': 6, 'CD2': 7, 'CE1': 8, 'CE2': 9},
}

ATOM14_AMBIG_TABLE = torch.zeros((21, 14)).long()
for res_name in residue_constants.residue_atoms.keys():
    res_letter = residue_constants.restype_3to1[res_name]
    res_idx = residue_constants.HHBLITS_AA_TO_ID[res_letter]

for res_name in residue_constants.residue_atom_renaming_swaps.keys():
    res_letter = residue_constants.restype_3to1[res_name]
    res_idx = residue_constants.HHBLITS_AA_TO_ID[res_letter]

    for swap_a in residue_constants.residue_atom_renaming_swaps[res_name].keys():
        swap_b = residue_constants.residue_atom_renaming_swaps[res_name][swap_a]
        swap_a_idx = residue_atom_renaming_swaps_idx[res_name][swap_a]
        swap_b_idx = residue_atom_renaming_swaps_idx[res_name][swap_b]

        ATOM14_AMBIG_TABLE[res_idx, swap_a_idx] = 1
        ATOM14_AMBIG_TABLE[res_idx, swap_b_idx] = 1


def get_rigid_groups(aatype, all_atom_positions, all_atom_mask):
    result = all_atom.atom37_to_frames(aatype, all_atom_positions, all_atom_mask)

    backbone_frame = result['rigidgroups_gt_frames'][:, 0, :]
    backbone_trans = backbone_frame[:, 9:]
    backbone_rots = backbone_frame[:, :9].view([-1, 3, 3])

    quat = quat_affine.rot_to_quat(backbone_rots, True)
    backbone_affine_tensor = torch.cat([quat, backbone_trans], dim=-1)
    result['backbone_affine_tensor'] = backbone_affine_tensor

    torsion_angles_reslut = all_atom.atom37_to_torsion_angles(
        aatype.unsqueeze(0), all_atom_positions.unsqueeze(0), all_atom_mask.unsqueeze(0)
    )
    result.update(torsion_angles_reslut)
    result['chi_angles'] = torsion_angles_reslut['torsion_angles_sin_cos'][:, :, 3:, :]
    result['chi_mask'] = torsion_angles_reslut['torsion_angles_mask'][:, :, 3:]
    result['alt_chi_angles'] = torsion_angles_reslut['alt_torsion_angles_sin_cos'][:, :, 3:, :]

    return result




def get_atom14_info(aatype, all_atom_positions, all_atom_mask, residx_atom14_to_atom37, atom14_atom_exists):
    batch = {
        'residx_atom14_to_atom37': residx_atom14_to_atom37,
        'atom14_atom_exists': atom14_atom_exists,
    }

    atom37_data = torch.cat([all_atom_positions, all_atom_mask.unsqueeze(-1)], dim=-1)
    atom14_result = all_atom.atom37_to_atom14(atom37_data, batch)
    atom14_positions, atom14_mask = atom14_result[:, :, :3], atom14_result[:, :, -1]

    #atom14_alt_gt_positions = atom14_positions
    atom14_alt_gt_positions = atom14_positions.clone().detach()
    atom14_alt_mask = atom14_mask.clone().detach()
    for i in range(aatype.shape[0]):
        aatype_i = aatype[i]
        if aatype_i == 2:
            atom14_alt_gt_positions[i, 6] = atom14_positions[i, 7]
            atom14_alt_gt_positions[i, 7] = atom14_positions[i, 6]
            atom14_alt_mask[i, 6] = atom14_mask[i, 7]
            atom14_alt_mask[i, 7] = atom14_mask[i, 6]
        
        if aatype_i == 3:
            atom14_alt_gt_positions[i, 7] = atom14_positions[i, 8]
            atom14_alt_gt_positions[i, 8] = atom14_positions[i, 7]

            atom14_alt_mask[i, 7] = atom14_mask[i, 8]
            atom14_alt_mask[i, 8] = atom14_mask[i, 7]

        if aatype_i in [4, 19]:
            atom14_alt_gt_positions[i, 6] = atom14_positions[i, 7]
            atom14_alt_gt_positions[i, 7] = atom14_positions[i, 6]
            atom14_alt_gt_positions[i, 8] = atom14_positions[i, 9]
            atom14_alt_gt_positions[i, 9] = atom14_positions[i, 8]

            atom14_alt_mask[i, 6] = atom14_mask[i, 7]
            atom14_alt_mask[i, 7] = atom14_mask[i, 6]
            atom14_alt_mask[i, 8] = atom14_mask[i, 9]
            atom14_alt_mask[i, 9] = atom14_mask[i, 8]

    result = {
        'atom14_gt_positions':  atom14_positions,
        'atom14_gt_exists': atom14_mask,  # ?
        'atom14_atom_is_ambiguous': ATOM14_AMBIG_TABLE[aatype],
        'atom14_alt_gt_positions': atom14_alt_gt_positions,
        'atom14_alt_gt_exists': atom14_alt_mask
    }

    return result


def expand_and_tile(mat, N=4):
    tile_size = [N] + len(mat.shape) * [1]
    return mat[None].repeat(tile_size) 


def make_structure_features(mmcif_dir, pdb_code, chain, st=0, ed=99999):
    cif_path = f'{mmcif_dir}/{pdb_code}.cif'
    cif_string = templates._read_file(cif_path)
    parsing_result = mmcif_parsing.parse(
        file_id=pdb_code, mmcif_string=cif_string)
    
    all_atom_positions, all_atom_mask = templates._get_atom_positions(
            parsing_result.mmcif_object, chain, max_ca_ca_distance=150.0)
    sequence = parsing_result.mmcif_object.chain_to_seqres[chain]

    return {
        "all_atom_positions": all_atom_positions[st:ed],
        "all_atom_mask": all_atom_mask[st:ed],
        "sequence": sequence[st:ed]
    }


def make_target_features(features, resolution, num_recycle=4, use_clamped_fape=True):
    # num_recycle = features['num_recycle'] + 1
    device = features['aatype'].device
    pseudo_beta, pseudo_beta_mask = data_transforms.pseudo_beta_fn(
        features['aatype'], features['all_atom_positions'], features['all_atom_mask']
    )

    features['pseudo_beta'] = pseudo_beta
    features['pseudo_beta_mask'] = pseudo_beta_mask
    features['resolution'] = torch.FloatTensor([resolution] * num_recycle).to(device)
    frame_results = get_rigid_groups(
            features['aatype'][0],
            features['all_atom_positions'][0],
            features['all_atom_mask'][0],
        )

    # backbone frame
    features['backbone_affine_tensor'] = expand_and_tile(frame_results['backbone_affine_tensor'], num_recycle)
    features['backbone_affine_mask'] = features['seq_mask']* (features['all_atom_mask'].sum(-1)>0)

    # 8 rigid groups of frames
    features['rigidgroups_gt_frames'] = expand_and_tile(frame_results['rigidgroups_gt_frames'], num_recycle)
    features['rigidgroups_alt_gt_frames']  = expand_and_tile(frame_results['rigidgroups_alt_gt_frames'], num_recycle)
    features['rigidgroups_gt_exists']  = expand_and_tile(frame_results['rigidgroups_gt_exists'], num_recycle)
    features['rigidgroups_group_exists']  = expand_and_tile(frame_results['rigidgroups_group_exists'], num_recycle)
    features['rigidgroups_group_is_ambiguous']  = expand_and_tile(frame_results['rigidgroups_group_is_ambiguous'], num_recycle)

    features['use_clamped_fape'] = torch.BoolTensor([use_clamped_fape]*num_recycle).to(device)

    features['chi_mask'] = expand_and_tile(frame_results['chi_mask'][0], num_recycle)
    features['chi_angles'] = expand_and_tile(frame_results['chi_angles'][0], num_recycle)
    features['torsion_angles_sin_cos'] = expand_and_tile(frame_results['torsion_angles_sin_cos'][0], num_recycle)
    features['alt_torsion_angles_sin_cos'] = expand_and_tile(frame_results['alt_torsion_angles_sin_cos'][0], num_recycle)
    features['torsion_angles_mask'] = expand_and_tile(frame_results['torsion_angles_mask'][0], num_recycle)

    # get atom14 coordinates
    atom14_result = get_atom14_info(
        features['aatype'][0],
        features['all_atom_positions'][0],
        features['all_atom_mask'][0],
        features['residx_atom14_to_atom37'][0],
        features['atom14_atom_exists'][0]
    )

    features['atom14_gt_positions'] = expand_and_tile(atom14_result['atom14_gt_positions'], num_recycle)
    features['atom14_alt_gt_positions'] = expand_and_tile(atom14_result['atom14_alt_gt_positions'], num_recycle)
    features['atom14_atom_is_ambiguous'] = expand_and_tile(atom14_result['atom14_atom_is_ambiguous'], num_recycle)
    features['atom14_gt_exists'] = expand_and_tile(atom14_result['atom14_gt_exists'], num_recycle)
    features['atom14_alt_gt_exists'] = expand_and_tile(atom14_result['atom14_alt_gt_exists'], num_recycle)
    features['seq_mask'] = features['seq_mask'].float()

    return features 