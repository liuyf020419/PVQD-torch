import os
import torch
import logging
import numpy as np
import random
import math
import collections
from .dataset import BaseDataset

import torch
from torch.utils import data
import torch.nn.functional as F

# from .data_transform import data_to_rigid_groups, perturb_rigid, sequence_mask
from .convert_aatype import convert_to_af_aatype

import sys
sys.path.append("protdiff/models")
from folding_af2.all_atom import atom37_to_frames
from folding_af2.common import residue_constants
from folding_af2.quat_affine import QuatAffine, quat_multiply, apply_rot_to_vec, quat_to_rot
from folding_af2.r3 import rigids_to_quataffine_m, rigids_from_tensor_flat12
# from alphafold import all_atom
from protein_utils import rigid, backbone, protein_cath
from protein_geom_utils import generate_pair_from_pos
from .data_transform import make_SS_condition, ss_letter2id

sys.path.append("/train14/superbrain/yfliu25/SCUBA_run/local_sd")
from protein_map_gen import FastPoteinParser
from protein_coord_parser import PoteinCoordsParser


sys.path.append("pdb_utils")
from pysketch import gen_peptides_ref_native_peptides, SS3_num_to_name
from calc_dssp import get_feature_from_dssp, preprocess_dssp_df, ENCODESS32NUM

logger = logging.getLogger(__name__)


restypes = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V'
]
restype_order = {restype: i for i, restype in enumerate(restypes)}
restype_num = len(restypes)  # := 20.
unk_restype_index = restype_num  # Catch-all index for unknown restypes.

restypes_with_x = restypes + ['X']
restype_order_with_x = {restype: i for i, restype in enumerate(restypes_with_x)}
idx_to_restype_with_x = {i: restype for i, restype in enumerate(restypes_with_x)}

mpnnalphabet = 'ACDEFGHIKLMNPQRSTVWYX'

STD_RIGID_COORD = torch.FloatTensor(
    [[-0.525, 1.363, 0.000],
    [0.000, 0.000, 0.000],
    [1.526, -0.000, -0.000],
    [0.627, 1.062, 0.000]]
)


def mask_chain_idx_coords_dict(coords_dict, mask_str):
    multichain_length_dict = coords_dict['multichain_length_dict']
    sequence = coords_dict['sequence']
    sstype = coords_dict['sstype']
    merged_coords = coords_dict['multichain_merged_coords'] # L, 14, 3
    chain_idx_list = np.array(coords_dict['merged_chain_label']).astype(np.int32)
    pdbresID = coords_dict['pdbresID']
    chain_list = list(map( int, mask_str.split('_') ))
    
    new_length_dict = {}
    new_pdbresID = {}
    new_chain_idx = []
    new_sequence = []
    new_sstype = []
    new_merged_coords = []
    reduced_chain_idx = []

    for chain_idx, chain_info in enumerate(multichain_length_dict.items()):
        if (chain_idx in chain_list):
            chain_unmask = chain_idx_list == chain_idx
            chain_str = chain_info[0]
            chain_len = chain_info[1]
            new_length_dict.update({chain_str: chain_len})
            new_pdbresID.update({chain_str: pdbresID[chain_str]})

            new_chain_idx.append(chain_idx_list[chain_unmask])
            new_sequence.append(np.array(list(sequence))[chain_unmask])
            new_sstype.append(sstype[chain_unmask])
            new_merged_coords.append(merged_coords[chain_unmask])
            reduced_chain_idx.append(chain_idx)

    masked_chain_dict = {
        'multichain_merged_coords': np.concatenate(new_merged_coords),
        'multichain_length_dict': new_length_dict,
        'sequence': ''.join(np.concatenate(new_sequence).tolist()),
        'sstype': np.concatenate(new_sstype),
        'pdbresID': new_pdbresID,
        'merged_chain_label': np.concatenate(new_chain_idx).tolist(),
        'reduced_chain_idx': reduced_chain_idx
    }

    return masked_chain_dict


class ProtDiffDataset(BaseDataset):
    def __init__(self, config, data_list, train=True, pdbroot=None, \
        noising_mode=None, validate=False, permute_data=False, random_seed=None, multichain_inference=False, split_chain=True):
        super().__init__()
        self.data_list= data_list
        self.config = config.model
        self.config_data = config.data
        self.global_config = self.config.global_config
        self.train_mode = train
        self.validate = validate
        self.multichain_inference = multichain_inference

        if self.train_mode:
            self.monomer_dataroot = config.data.monomer_data_path
            self.complex_dataroot = config.data.complex_data_path
            self.structure_root = config.data.base_path
        else:
            self.dataroot = pdbroot
            self.noising_mode = noising_mode

        self.protein_list = []
        self._epoch = 0
        if validate:
            self.max_len = 10000
        else:
            self.max_len = self.global_config.max_len
        self.gap_between_chain = self.config_data.common.gap_between_chain
        # import pdb; pdb.set_trace()
        if (self.train_mode and self.validate):
            self.split_chain_p = float(split_chain)
        else:
            self.split_chain_p = self.config_data.common.split_chain_p
        self.split_chain_pos_center_var = self.config_data.common.pos_center_var

        with open(data_list, 'r') as f:
            for line in f:
                if self.train_mode:
                    self.protein_list.append( line.strip().split() )
                else:
                    line_split = line.strip().split("_")
                    name = line_split[0]
                    if not self.multichain_inference:
                        chain = line_split[1]
                        self.protein_list.append((name, chain))
                    else:
                        self.protein_list.append(name)


        logger.info(f'list size: {len(self.protein_list)}')

        if permute_data:
            if random_seed is None:
                np.random.seed(None)
            else:
                assert isinstance(random_seed, int)
                np.random.seed(random_seed)
            np.random.shuffle(self.protein_list)


    def __len__(self):
        return len(self.protein_list)

    def data_sizes(self):
        return [l[1] for l in self.protein_list]
    
    def reset_data(self, epoch):
        self._epoch = epoch

    def __getitem__(self, item:int):
        if self.train_mode:
            try:
                data_mode, protein, chain_mask_str = self.protein_list[item]
                if data_mode == 'monomer':
                    dataroot = self.monomer_dataroot
                elif data_mode == 'complex':
                    dataroot = self.complex_dataroot
                test_data_file = os.path.join(f'{dataroot}/{protein[1:3]}/{protein}.npy')
                test_tmpdata = np.load(test_data_file, allow_pickle=True)

                loss_mask = False
            except:
                data_mode, protein, chain_mask_str = self.protein_list[0]
                if data_mode == 'monomer':
                    dataroot = self.monomer_dataroot
                elif data_mode == 'complex':
                    dataroot = self.complex_dataroot
                loss_mask = True

            try:
                data_file = os.path.join(f'{dataroot}/{protein[1:3]}/{protein}.npy')
                tmpdata = np.load(data_file, allow_pickle=True).item()
                coords_dict = tmpdata['coords']
                protein_len = len(coords_dict['sequence'])

                if data_mode == 'monomer':
                    multichain_length_dict = {'A': protein_len}
                    merged_coords = coords_dict['merged_coords']
                    raw_pdb_res_id = {'A': coords_dict['pdbresID']}
                    # merged_chain_label = np.zeros(protein_len).tolist()
                else:
                    if (chain_mask_str != 'ALL'):
                       coords_dict = mask_chain_idx_coords_dict(coords_dict, chain_mask_str)
                       protein_len = len(coords_dict['sequence'])
                    multichain_length_dict = coords_dict['multichain_length_dict']
                    merged_coords = coords_dict['multichain_merged_coords']
                    raw_pdb_res_id = coords_dict['pdbresID']
                    # merged_chain_label = coords_dict['merged_chain_label']

                chains_len = list(multichain_length_dict.values())
                raw_single_res_id_list = [ chain_resid for chain_resid in raw_pdb_res_id.values()]
                merged_sequence_str = coords_dict['sequence']
                
                data_dict = {}
                # import pdb; pdb.set_trace()
                encode_split_chain = 0.
                if len(chains_len) > 1:
                    if self.split_chain_p > np.random.rand(1)[0]:
                        encode_split_chain = 1
                data_dict['encode_split_chain'] = torch.tensor([encode_split_chain])

                if self.config_data.common.pad_data:
                    if (self.train_mode and self.validate):
                        crop_max_len = protein_len
                        max_len = protein_len
                    else:
                        crop_max_len = self.max_len
                        max_len = self.max_len
                else:
                    if protein_len >= self.max_len:
                        max_len = crop_max_len = self.max_len
                    else:
                        max_len = crop_max_len = protein_len

                if np.random.rand(1)[0] > self.config_data.common.spatial_crop_p:
                    chain_mask = self.crop_contiguous([protein_len], crop_max_len)
                else:
                    chain_mask = self.crop_spatial(
                        merged_coords, crop_max_len)

                if (protein_len == len(chain_mask)):
                    pass
                elif (protein_len > len(chain_mask)):
                    mask_pad_num = protein_len - len(chain_mask)
                    chain_mask = np.concatenate([chain_mask, np.zeros((mask_pad_num, ))])
                elif (protein_len < len(chain_mask)):
                    mask_crop_num = len(chain_mask) - protein_len
                    chain_mask = chain_mask[:-mask_crop_num]

                # make crd
                preprocess_crd_dict={
                    'multichain_length_dict': multichain_length_dict,
                    'multichain_merged_coords': merged_coords,
                    'sequence': merged_sequence_str,
                    'sstype': coords_dict['sstype']
                }
                self.merge_pos_frame_data(data_dict, preprocess_crd_dict, torch.from_numpy(chain_mask))
                # make possition embedding
                merged_pdbresID = self.make_multichain_single_res_idx(raw_single_res_id_list, self.gap_between_chain)
                chain_rel_pos_dict = self.add_assembly_feature(chains_len, merged_pdbresID, merged_sequence_str)
                self.make_multichains_rel_pos(data_dict, chain_rel_pos_dict)

                data_dict["loss_mask"] = torch.tensor([loss_mask])
                data_dict['pdbname'] = protein
                data_dict['max_len'] = max_len
                data_dict['chain_mask_str'] = chain_mask_str
                try:
                    data_dict['reduced_chain_idx'] = coords_dict['reduced_chain_idx']
                except:
                    pass
                data_dict = self.crop_data(data_dict, torch.from_numpy(chain_mask))

            except:
                data_mode, protein, chain_mask_str = self.protein_list[0]
                if data_mode == 'monomer':
                    dataroot = self.monomer_dataroot
                elif data_mode == 'complex':
                    dataroot = self.complex_dataroot
                loss_mask = True

                data_file = os.path.join(f'{dataroot}/{protein[1:3]}/{protein}.npy')
                tmpdata = np.load(data_file, allow_pickle=True).item()
                coords_dict = tmpdata['coords']
                # monomer_len = coords_dict['merged_coords'].shape[0]
                protein_len = len(coords_dict['sequence'])

                if data_mode == 'monomer':
                    multichain_length_dict = {'A': protein_len}
                    merged_coords = coords_dict['merged_coords']
                    raw_pdb_res_id = {'A': coords_dict['pdbresID']}
                    # merged_chain_label = np.zeros(protein_len).tolist()
                else:
                    if (chain_mask_str != 'ALL'):
                       coords_dict = mask_chain_idx_coords_dict(coords_dict, chain_mask_str)
                       protein_len = len(coords_dict['sequence'])
                    multichain_length_dict = coords_dict['multichain_length_dict']
                    merged_coords = coords_dict['multichain_merged_coords']
                    raw_pdb_res_id = coords_dict['pdbresID']
                    # merged_chain_label = coords_dict['merged_chain_label']

                chains_len = list(multichain_length_dict.values())
                raw_single_res_id_list = [ chain_resid for chain_resid in raw_pdb_res_id.values()]
                merged_sequence_str = coords_dict['sequence']
                
                data_dict = {}
                encode_split_chain = 0.
                if len(chains_len) > 1:
                    if self.split_chain_p > np.random.rand(1)[0]:
                        encode_split_chain = 1
                data_dict['encode_split_chain'] = torch.tensor([encode_split_chain])

                if self.config_data.common.pad_data:
                    if (self.train_mode and self.validate):
                        crop_max_len = protein_len
                        max_len = protein_len
                    else:
                        crop_max_len = self.max_len
                        max_len = self.max_len
                else:
                    if protein_len >= self.max_len:
                        max_len = crop_max_len = self.max_len
                    else:
                        max_len = crop_max_len = protein_len

                if np.random.rand(1)[0] > self.config_data.common.spatial_crop_p:
                    chain_mask = self.crop_contiguous([protein_len], crop_max_len)
                else:
                    chain_mask = self.crop_spatial(
                        merged_coords, crop_max_len)

                if (protein_len == len(chain_mask)):
                    pass
                elif (protein_len > len(chain_mask)):
                    mask_pad_num = protein_len - len(chain_mask)
                    chain_mask = np.concatenate([chain_mask, np.zeros((mask_pad_num, ))])
                elif (protein_len < len(chain_mask)):
                    mask_crop_num = len(chain_mask) - protein_len
                    chain_mask = chain_mask[:-mask_crop_num]

                # make crd
                preprocess_crd_dict={
                    'multichain_length_dict': multichain_length_dict,
                    'multichain_merged_coords': merged_coords,
                    'sequence': merged_sequence_str,
                    'sstype': coords_dict['sstype']
                }
                self.merge_pos_frame_data(data_dict, preprocess_crd_dict, torch.from_numpy(chain_mask))
                # make possition embedding
                merged_pdbresID = self.make_multichain_single_res_idx(raw_single_res_id_list, self.gap_between_chain)
                chain_rel_pos_dict = self.add_assembly_feature(chains_len, merged_pdbresID, merged_sequence_str)
                self.make_multichains_rel_pos(data_dict, chain_rel_pos_dict)

                data_dict["loss_mask"] = torch.tensor([loss_mask])
                data_dict['pdbname'] = protein
                data_dict['max_len'] = max_len
                data_dict['chain_mask_str'] = chain_mask_str
                try:
                    data_dict['reduced_chain_idx'] = coords_dict['reduced_chain_idx']
                except:
                    pass
                data_dict = self.crop_data(data_dict, torch.from_numpy(chain_mask))

            return data_dict

        else:
            if self.multichain_inference:
                protein = self.protein_list[item]
                pdbfile = f'{self.dataroot}/{protein}'
                data_dict = self.make_from_pdb_file(pdbfile)
            else:
                protein, chain = self.protein_list[item]
                pdbfile = f'{self.dataroot}/{protein}'
                data_dict = self.make_from_pdb_file(pdbfile, chain)

            resrange = (-self.global_config.pair_res_range[1], self.global_config.pair_res_range[1])
            resmask_num = self.global_config.pair_res_range[1] + 1
            chainrange = (-self.global_config.pair_chain_range[1], self.global_config.pair_chain_range[1])
            chainmask_num = self.global_config.pair_chain_range[1] + 1
            
            relpdb_residx = np.array(data_dict['pdbresID']) - np.array(data_dict['pdbresID']).min()
            self.get_position_embedding(data_dict, relpdb_residx=relpdb_residx, 
                                        enlarge_gap=self.enlarge_gap, resrange=resrange,
                                        resmask_num=resmask_num, chainrange=chainrange, chainmask_num=chainmask_num)
            
            data_dict['pdbname'] = protein
            data_dict['pdbresID'] = torch.from_numpy(data_dict['pdbresID'])
            data_dict["loss_mask"] = torch.tensor([False])

            return data_dict


    def crop_contiguous(self, chains_len, max_num_res):
        n_added = 0
        n_remaining = sum(chains_len)
        chains_mask = []
        for k in range(len(chains_len)):
            cur_chain_mask = np.zeros((chains_len[k], ))
            n_remaining = n_remaining - chains_len[k]
            crop_size_max = min(max_num_res - n_added, chains_len[k])
            crop_size_min = min(chains_len[k], max(0, max_num_res - (n_added + n_remaining)))
            # import pdb; pdb.set_trace()
            crop_size = np.random.randint(crop_size_min, crop_size_max+1, 1)[0]
            n_added += crop_size
            crop_start_res_id = np.random.randint(0, chains_len[k]-crop_size+1, 1)[0]
            cur_chain_mask[crop_start_res_id: crop_start_res_id+crop_size] = np.ones((crop_size, ))
            chains_mask.append(cur_chain_mask)

        all_chain_mask = np.concatenate(chains_mask)
        return all_chain_mask


    def crop_spatial(self, merged_true_coords, max_num_res):
        # assert len(true_coords_list[0].shape) == 3 # [L, n_res, 3]q
        assert len(merged_true_coords.shape) == 3 # [L, n_res, 3]
        # merged_true_coords = np.concatenate(true_coords_list)
        merged_res_num = merged_true_coords.shape[0]

        selected_center_res_idx = np.random.randint(0, merged_res_num)

        selected_interface_res_coord = merged_true_coords[selected_center_res_idx]
        knn_ca_dist = np.sqrt(np.sum(np.square(selected_interface_res_coord[None][:, 1] - merged_true_coords[:, 1]), -1) + 1e-10)
        knn_res_idx = np.argsort(knn_ca_dist)[: max_num_res]
        chains_mask = np.zeros((merged_res_num, ))
        chains_mask[knn_res_idx] = np.ones((np.minimum(max_num_res, merged_res_num), ))

        return chains_mask


    def clamp_data(self, data_dict):
        new_data = {}
        max_len = self.max_len

        for name in data_dict.keys():
            if name in ['loss_mask', 'pdbname', 'noising_mode_idx', 'cath_architecture']:
                new_data[name] = data_dict[name]
            elif name in ['len']:
                new_data[name] = torch.LongTensor([max_len])
            elif name in ['traj_pos', 'gt_pos', 'traj_backbone_frame', 'gt_backbone_frame', 'traj_backbone_frame_ss', 'traj_pos_ss']:
                new_data[name] = data_dict[name][:max_len]
            elif name in ['single_res_rel', 'aatype', 'single_ssedges', 'masked_FG_seq', 'sstype', 'norm_esm_single', 'unnorm_esm_single', 'esm_single_mask']:
                new_data[name] = data_dict[name][:max_len]
            elif name in ['pair_res_rel', 'pair_chain_rel', 'ss_adj_pair', 'masked_pair_map']:
                new_data[name] = data_dict[name][:max_len, :max_len]
            else:
                continue

        return new_data

    
    def crop_data(self, data_dict, multichain_mask):
        new_data = {}
        length = torch.sum(multichain_mask).item()
        multichain_mask = multichain_mask.bool()

        for name in data_dict.keys():
            if name in [
                'loss_mask', 'pdbname', 'noising_mode_idx', 'cath_architecture', 'encode_split_chain', 'reduced_chain_idx', 'chain_mask_str'
                ]:
                new_data[name] = data_dict[name]
            elif name in ['len']:
                new_data[name] = torch.LongTensor([length])
            elif name in ['max_len']:
                new_data[name] = torch.LongTensor([data_dict[name]])
            elif name in [
                'traj_pos', 'gt_pos', 'traj_backbone_frame', 'gt_backbone_frame', 
                'traj_backbone_frame_ss', 'traj_pos_ss'
                ]:
                new_data[name] = data_dict[name][multichain_mask]
            elif name in [
                'single_res_rel', 'single_chain_rel', 'aatype', 'mpnn_aatype', 'single_ssedges', 'masked_FG_seq', 
                'sstype', 'norm_esm_single', 'unnorm_esm_single', 'esm_single_mask', 'chain_idx', 'entity_idx',
                'sd_replaced_region'
                ]:
                new_data[name] = data_dict[name][multichain_mask]
            elif name in [
                'pair_res_rel', 'pair_chain_rel', 'ss_adj_pair', 'masked_pair_map', 
                'pair_res_idx', 'pair_same_entity', 'pair_chain_idx', 'pair_same_chain'
                ]:
                new_data[name] = data_dict[name][multichain_mask][:, multichain_mask]
            else:
                continue

        return new_data

    
    def merge_pos_frame_data(self, data_dict: dict, preprocess_crd_dict: dict, chain_mask):
        chain_mask = chain_mask.bool()
        multichain_length_dict = preprocess_crd_dict['multichain_length_dict']
        gt_pos = torch.from_numpy(preprocess_crd_dict['multichain_merged_coords']).float()
        pos_center = torch.cat([gt_pos[chain_mask][:, 1]]).mean(0)
        gt_pos = gt_pos - pos_center
        gt_backbone_frame = get_quataffine(gt_pos)

        is_nomomer = len(multichain_length_dict) == 1
        if ((not is_nomomer) and (data_dict['encode_split_chain'][0].item() == 1)):
            traj_chain_coords_list, traj_chain_frame_list = [], []
            chain_length_summed = 0
            for chain_length in multichain_length_dict.values():
                start_index = chain_length_summed
                chain_length_summed = chain_length + chain_length_summed
                end_index = chain_length_summed

                chain_crd = gt_pos[start_index:end_index]
                chain_pos_center = torch.cat([chain_crd[:, 1]]).mean(0)
                chain_crd = chain_crd - chain_pos_center + (torch.rand(3) * self.split_chain_pos_center_var)[None, None, :]
                chain_backbone_frame = get_quataffine(chain_crd)
                traj_chain_coords_list.append(chain_crd)
                traj_chain_frame_list.append(chain_backbone_frame)
                
            traj_pos = torch.cat(traj_chain_coords_list)
            traj_backbone_frame = torch.cat(traj_chain_frame_list)
        else:
            traj_pos = gt_pos
            traj_backbone_frame = gt_backbone_frame

        aatype = [restype_order_with_x[aa] for aa in preprocess_crd_dict["sequence"]]
        sstype = torch.from_numpy(preprocess_crd_dict['sstype']).long()

        data_dict["aatype"] = torch.LongTensor(aatype)
        mpnn_aatype = torch.LongTensor([
            mpnnalphabet.index(aa) for aa in preprocess_crd_dict["sequence"]])
        data_dict['mpnn_aatype'] = mpnn_aatype
        data_dict["len"] = torch.LongTensor([len(aatype)])
        
        data_dict["gt_pos"] = gt_pos
        data_dict["gt_backbone_frame"] = gt_backbone_frame
        data_dict["traj_pos"] = traj_pos
        data_dict["traj_backbone_frame"] = traj_backbone_frame

        data_dict['sstype'] = sstype[:, 0]
        


    def make_from_pdb_file(self, poteinfile, chain=None):
        data_dict = {}
        if not self.multichain_inference:
            assert chain is not None
            PDBparser = PoteinCoordsParser(poteinfile, chain)
            gt_pos = torch.from_numpy(PDBparser.chain_main_crd_array.reshape(-1,5,3)).float()

            pos_center = torch.cat([gt_pos]).mean(0)
            gt_pos = gt_pos - pos_center
            gt_backbone_frame = get_quataffine(gt_pos)
            sequence = PDBparser.get_sequence(chain)
            data_dict['pdbresID'] = np.array(list(PDBparser.get_pdbresID2absID(chain).keys()))
        else:
            PDBparser = PoteinCoordsParser(poteinfile)
            chain_list = list(PDBparser.chain_crd_dicts.keys())
            # chain_list = ['B']
            chain_sequence_list, chain_coords_list, chain_frame_list, merged_pdbresID, merged_chainID = [], [], [], [], []
            for c_idx, chain in enumerate(chain_list):
                chain_crd = torch.from_numpy(PDBparser.get_main_crd_array(chain).reshape(-1,5,3)).float()
                sequence = PDBparser.get_sequence(chain)
                chain_sequence_list.append(sequence)
                chain_pos_center = torch.cat([chain_crd[:, 1]]).mean(0)
                chain_crd = chain_crd - chain_pos_center + (torch.rand(3) * 100)[None, None, :]
                chain_backbone_frame = get_quataffine(chain_crd)
                chain_coords_list.append(chain_crd)
                chain_frame_list.append(chain_backbone_frame)
                raw_pdbres_idx = np.array(list(PDBparser.get_pdbresID2absID(chain).keys()))
                if c_idx == 0:
                    new_pdbres_idx = raw_pdbres_idx - raw_pdbres_idx[0]
                else:
                    new_pdbres_idx = raw_pdbres_idx - raw_pdbres_idx[0] + merged_pdbresID[-1][-1] + 1
                merged_pdbresID.append(new_pdbres_idx)
                merged_chainID.append(torch.ones(len(sequence)) * c_idx)

            sequence = ''.join(chain_sequence_list)
            gt_pos = torch.cat(chain_coords_list)
            gt_backbone_frame = torch.cat(chain_frame_list)
            data_dict['pdbresID'] = np.concatenate(merged_pdbresID)
            data_dict['chain_idx'] = torch.cat(merged_chainID)
        
        aatype = [restype_order_with_x[aa] for aa in sequence]
        data_dict["aatype"] = torch.LongTensor(aatype)
        mpnn_aatype = torch.LongTensor([
            mpnnalphabet.index(aa) for aa in sequence])
        data_dict['mpnn_aatype'] = mpnn_aatype
        data_dict["len"] = torch.LongTensor([len(aatype)])
        
        data_dict["gt_pos"] = gt_pos
        data_dict["gt_backbone_frame"] = gt_backbone_frame


        return data_dict



    def get_position_embedding(self, data_dict, relpdb_residx, resrange=(-32, 32), resmask_num=33, 
                                    chainrange=(-4, 4), chainmask_num=5, enlarge_gap=True, gap_size=100):

        split_idx = np.arange(len(relpdb_residx))[np.append(np.diff(relpdb_residx) != 1, False)] + 1
        # last chain
        chain_num = len(split_idx) + 1
        chain_lens = np.diff(np.append(np.concatenate([[0], split_idx]), len(relpdb_residx) ))

        if enlarge_gap:
            res_rel_idx = []
            for idx, chain_len in enumerate(chain_lens):
                if idx != 0:
                    res_rel_idx.extend(np.arange(chain_len) + res_rel_idx[-1] + gap_size)
                else:
                    res_rel_idx.extend(np.arange(chain_len))

            data_dict["single_res_rel"] = torch.LongTensor(res_rel_idx)

        else:
            single_part_res_rel_idx = np.concatenate([np.arange(chain_len) for chain_len in chain_lens])
            single_all_chain_rel_idx = np.concatenate([np.ones(chain_len[1], dtype=np.int32) * chain_len[0] \
                                                        for chain_len in enumerate(chain_lens)])

            single_all_res_rel_idx = relpdb_residx - relpdb_residx[0]
            data_dict["single_all_res_rel"] = torch.from_numpy(single_all_res_rel_idx)
            data_dict["single_part_res_rel"] = torch.from_numpy(single_part_res_rel_idx)
            data_dict["single_all_chain_rel"] = torch.from_numpy(single_all_chain_rel_idx)


        pair_res_rel_idx = relpdb_residx[:, None] - relpdb_residx

        unclip_single_chain_rel_idx = np.repeat(np.arange(chain_num), chain_lens)
        pair_chain_rel_idx = unclip_single_chain_rel_idx[:, None] - unclip_single_chain_rel_idx
        
        pair_res_rel_idx = np.where(np.any(np.stack([pair_res_rel_idx > resrange[1], 
                                pair_res_rel_idx < resrange[0]]), 0), resmask_num, pair_res_rel_idx)

        pair_chain_rel_idx = np.where(np.any(np.stack([pair_chain_rel_idx > chainrange[1], 
                                pair_chain_rel_idx < chainrange[0]]), 0), chainmask_num, pair_chain_rel_idx)

        data_dict["pair_res_rel"] = torch.from_numpy(pair_res_rel_idx.astype(np.int64)) - resrange[0]
        data_dict["pair_chain_rel"] = torch.from_numpy(pair_chain_rel_idx.astype(np.int64)) - chainrange[0]


    def make_multichain_single_res_idx(self, raw_single_res_id_list, gap_between_chain):
        merged_single_res_idx = []
        for c_idx, raw_c_pdbres_idx in enumerate(raw_single_res_id_list):
            raw_c_pdbres_idx = np.array(raw_c_pdbres_idx)
            if c_idx == 0:
                new_c_pdbres_idx = raw_c_pdbres_idx - raw_c_pdbres_idx[0]
            else:
                new_c_pdbres_idx = raw_c_pdbres_idx - raw_c_pdbres_idx[0] + merged_single_res_idx[-1][-1] + gap_between_chain
            merged_single_res_idx.append(new_c_pdbres_idx)

        merged_single_res_idx = torch.from_numpy( np.concatenate(merged_single_res_idx) ).long()

        return merged_single_res_idx


    def add_assembly_feature(self, chain_lens, merged_pdbresID, seq_str):
        rel_all_chain_features = {}
        seq_to_entity_id = {}
        grouped_chains_length = collections.defaultdict(list)
        chain_length_summed = 0
        for chain_len in chain_lens:
            start_index = chain_length_summed
            chain_length_summed += int(chain_len)
            end_index = chain_length_summed
            seq = seq_str[start_index: end_index]
            if seq not in seq_to_entity_id:
                seq_to_entity_id[seq] = len(seq_to_entity_id) + 1
            grouped_chains_length[seq_to_entity_id[seq]].append(chain_len)

        asym_id_list, sym_id_list, entity_id_list, num_sym_list = [], [], [], []
        chain_id = 0
        for entity_id, group_chain_features in grouped_chains_length.items():
            num_sym = len(group_chain_features)  # zy
            for sym_id, seq_length in enumerate(group_chain_features, start=1):
                asym_id_list.append(chain_id * torch.ones(seq_length)) # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
                sym_id_list.append(sym_id * torch.ones(seq_length)) # [1,2,3,1,2,3,1,2,3,1,2,3,1,2,3]
                entity_id_list.append(entity_id * torch.ones(seq_length)) # [1,1,1,2,2,2,3,3,3,4,4,4,5,5,5]
                num_sym_list.append(num_sym * torch.ones(seq_length)) # [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
                chain_id += 1

        rel_all_chain_features['asym_id'] = torch.cat(asym_id_list)
        rel_all_chain_features['sym_id'] = torch.cat(sym_id_list)
        rel_all_chain_features['entity_id'] = torch.cat(entity_id_list)
        rel_all_chain_features['num_sym'] = torch.cat(num_sym_list)
        rel_all_chain_features['res_id'] = merged_pdbresID

        return rel_all_chain_features


    
    def make_multichains_rel_pos(self, data_dict: str, chain_rel_pos_dict: str, rmax=32, smax=5):
        # pair pos
        diff_aym_id = (chain_rel_pos_dict['asym_id'][None, :] - chain_rel_pos_dict['asym_id'][:, None])
        diff_res_id = (chain_rel_pos_dict['res_id'][None, :] - chain_rel_pos_dict['res_id'][:, None])
        diff_sym_id = (chain_rel_pos_dict['sym_id'][None, :] - chain_rel_pos_dict['sym_id'][:, None])
        diff_entity_id = (chain_rel_pos_dict['entity_id'][None, :] - chain_rel_pos_dict['entity_id'][:, None])

        clamp_res_id = torch.clamp(diff_res_id+rmax, min=0, max=2*rmax)
        pair_res_idx = torch.where(diff_aym_id.long() == 0, clamp_res_id.long(), 2*rmax+1) # 2*rmax + 2

        same_chain = (chain_rel_pos_dict['asym_id'][None, :] == chain_rel_pos_dict['asym_id'][:, None]).long()
        same_entity = (chain_rel_pos_dict['entity_id'][None, :] == chain_rel_pos_dict['entity_id'][:, None]).long() # 2 + 1

        clamp_sym_id = torch.clamp(diff_sym_id+smax, min=0, max=2*smax)
        pair_chain_idx = torch.where(diff_entity_id.long() == 0, clamp_sym_id.long(), 2*smax+1) # 2*smax + 2

        pair_rel_pos_dict = {
            'pair_res_idx': pair_res_idx,
            'pair_same_entity': same_entity,
            'pair_chain_idx': pair_chain_idx,
            'pair_same_chain': same_chain,
            'single_res_rel': chain_rel_pos_dict['res_id'],
            'chain_idx': chain_rel_pos_dict['asym_id'],
            'entity_idx': chain_rel_pos_dict['entity_id']-1
        }
        data_dict.update(pair_rel_pos_dict)


def get_quataffine(pos):
    assert len(pos.shape)
    nres, natoms, _ = pos.shape
    assert natoms == 5
    alanine_idx = residue_constants.restype_order_with_x["A"]
    aatype = torch.LongTensor([alanine_idx] * nres)
    all_atom_positions = F.pad(pos, (0,0,0,37-5), "constant", 0)
    all_atom_mask = torch.ones(nres, 37)
    frame_dict = atom37_to_frames(aatype, all_atom_positions, all_atom_mask)

    return frame_dict['rigidgroups_gt_frames']


def add_pseudo_c_beta_from_gly(pos):
    vec_ca = pos[:, 1]
    vec_n = pos[:, 0]
    vec_c = pos[:, 2]
    vec_o = pos[:, 3]
    b = vec_ca - vec_n
    c = vec_c - vec_ca
    a = torch.cross(b, c)
    vec_cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + vec_ca
    return torch.stack([vec_n, vec_ca, vec_c, vec_cb, vec_o]).permute(1,0,2)


def noising_coord(x0, noise_scale):
    noise = torch.zeros_like(x0)
    xt = x0 + noise_scale * noise
    return xt


def noising_quat(q0):
    batch_size = q0.shape[0]
    qT = rigid.rand_quat(q0.shape[:-1]).to(q0.device)
    alpha_t = torch.rand((batch_size, ), device=q0.device)
    qt = rigid.slert(q0, qT, alpha_t)
    return qt


def permute_between_ss_from_affine12(affine_flat12, sstype, ca_noise_scale, gt_pos):
    affine = rigids_to_quataffine_m(rigids_from_tensor_flat12(affine_flat12)).to_tensor()[:, 0]
    ca_pos = affine[..., 4:]
    ss3type = sstype[:, 0]
    ss_start_indexs = (torch.where((ss3type[1:] - ss3type[:-1]) != 0)[0] + 1).long()
    ss_start_indexs = torch.cat([torch.LongTensor([0]), ss_start_indexs])
    ss_end_indexs = torch.cat([ss_start_indexs[1:]-1, torch.LongTensor([len(ss3type)])])
    ss_lens = ss_start_indexs[1:] - ss_start_indexs[:-1]
    ss_lens = torch.cat([ss_lens, (len(ss3type) - ss_start_indexs[-1]).unsqueeze(0)])
    start_sstypes = torch.index_select(ss3type, 0, ss_start_indexs)

    if isinstance(ca_noise_scale, list):
        ca_noise_scale = np.random.uniform(ca_noise_scale[0], ca_noise_scale[1], 1)[0]

    traj_coords = []
    traj_flat12s = []
    for ss_idx, ss in enumerate(start_sstypes):
        ss_len = ss_lens[ss_idx]
        ss_start_index = ss_start_indexs[ss_idx]
        ss_end_index = ss_end_indexs[ss_idx]
        ss_mid_index = (ss_end_index-ss_start_index)//2
        # print(ss_len, ss)
        if ((ss_len > 2) and (ss != 1)):
            ss_rigid_x = ca_pos[ss_start_index]
            ss_rigid_y = ca_pos[ss_mid_index]
            ss_rigid_z = ca_pos[ss_end_index]
            ss_rigid = torch.cat([ss_rigid_x[None], ss_rigid_y[None], ss_rigid_z[None]])

            gt_ss_affine = rigid.pos_to_affine(ss_rigid[None])
            # traj_quat = noising_quat(gt_ss_affine[..., :4])
            traj_ss_quat = gt_ss_affine[..., :4]
            traj_ss_rot = rigid.quat_to_rot(traj_ss_quat[0])
            # traj_trans = noising_coord(gt_ss_affine[..., 4:], ca_noise_scale)[0]
            traj_ss_trans = gt_ss_affine[..., 4:][0]

            gt_affine_stack = affine[ss_start_index: ss_end_index+1]
            traj_quat = gt_affine_stack[..., :4]
            traj_trans = gt_affine_stack[..., 4:]
            # pseudo_aatype = torch.zeros((ss_len, ), dtype=torch.long, device=gt_affine_stack.device)
            traj_rot = rigid.quat_to_rot(traj_quat)

            traj_coord = backbone.backbone_frame_to_atom3_std(
                            torch.reshape(traj_rot, (-1, 3, 3)), 
                            torch.reshape(traj_trans, (-1, 3)),
                            atomnum=4
            )

            traj_coord = update_rigid_pos(traj_coord, traj_ss_trans, traj_ss_rot)
            # traj_coords.append(traj_coord)
            traj_coord = add_pseudo_c_beta_from_gly(traj_coord)
            traj_flat12 = get_quataffine(traj_coord)
            # print(traj_flat12.shape)
            traj_coords.append(traj_coord)
            traj_flat12s.append(traj_flat12)

        else:
            traj_coord = STD_RIGID_COORD[None].repeat(ss_len, 1, 1)
            traj_coords.append(add_pseudo_c_beta_from_gly(traj_coord))
            traj_flat12s.append(affine_flat12[ss_start_index: ss_end_index+1])

    traj_coords = torch.cat(traj_coords)
    traj_flat12s = torch.cat(traj_flat12s)

    return traj_coords, traj_flat12s



def permute_between_ss_from_pos(
    gt_pos, 
    sstype, 
    ca_noise_scale, 
    quat_noise_scale, 
    white_noise_scale, 
    sketch_data, 
    ss_mask_p_range, 
    loop_mask_p_range, 
    ss_wrong_p_range
    ):
    ca_pos = gt_pos[:, 1]
    ss3type = sstype[:, 0]
    ss_start_indexs = (torch.where((ss3type[1:] - ss3type[:-1]) != 0)[0] + 1).long()
    ss_start_indexs = torch.cat([torch.LongTensor([0]), ss_start_indexs])
    ss_end_indexs = torch.cat([ss_start_indexs[1:]-1, torch.LongTensor([len(ss3type)])])
    ss_lens = ss_start_indexs[1:] - ss_start_indexs[:-1]
    ss_lens = torch.cat([ss_lens, (len(ss3type) - ss_start_indexs[-1]).unsqueeze(0)])
    start_sstypes = torch.index_select(ss3type, 0, ss_start_indexs)

    if isinstance(ca_noise_scale, list):
        ca_noise_scale = np.random.uniform(ca_noise_scale[0], ca_noise_scale[1], 1)[0]
    assert isinstance(ss_mask_p_range, list)
    ss_mask_p = np.random.uniform(ss_mask_p_range[0], ss_mask_p_range[1], 1)[0]
    assert isinstance(loop_mask_p_range, list)
    loop_mask_p = np.random.uniform(loop_mask_p_range[0], loop_mask_p_range[1], 1)[0]
    assert isinstance(ss_wrong_p_range, list)
    ss_replace_p = np.random.uniform(ss_wrong_p_range[0], ss_wrong_p_range[1], 1)[0]

    traj_coords = []
    for ss_idx, ss in enumerate(start_sstypes):
        ss = ss.item()
        ss_len = ss_lens[ss_idx]
        ss_start_index = ss_start_indexs[ss_idx]
        ss_end_index = ss_end_indexs[ss_idx]
        gt_ss_pos = gt_pos[ss_start_index: ss_end_index+1]

        if ((ss_len > 2) and (ss != 1)):
            if np.random.rand(1)[0] > ss_mask_p:
                if np.random.rand(1)[0] < ss_replace_p:
                    if ss == 2:
                        ss = 0
                    else:
                        ss = 2
                ss_frame = rigid.rigid_from_3_points(gt_ss_pos[0, 1], gt_ss_pos[ss_len.item()//2, 1], gt_ss_pos[-1, 1])
                ss_quat = rigid.rot_to_quat(ss_frame['rot'])
                # traj_quat = updated_noising_quat(ss_quat[None], quat_noise_scale)
                qT = rigid.rand_quat(ss_quat.shape[:-1]).to(ss_quat.device)

                new_traj_rot = rigid.quat_to_rot(qT)

                updated_traj_trans = torch.randn(3) * ca_noise_scale
                
                if sketch_data:
                    sketch_ss_pos = add_pseudo_c_beta_from_gly(
                        torch.from_numpy(
                            gen_peptides_ref_native_peptides(gt_ss_pos.numpy()[:, :4], 
                            SS3_num_to_name[ss])).float())
                    traj_ss_pos = update_rigid_pos_new(sketch_ss_pos, updated_traj_trans, new_traj_rot)
                else:
                    traj_ss_pos = update_rigid_pos_new(gt_ss_pos, updated_traj_trans, new_traj_rot)
                traj_coords.append(traj_ss_pos)
            else:
                noising_quat = rigid.rand_quat([1, ss_len])
                noising_coord = torch.randn(1, ss_len, 3) * white_noise_scale
                noising_affine = torch.cat([noising_quat, noising_coord], -1)
                noising_pos = rigid.affine_to_pos(noising_affine.reshape(-1, 7)).reshape(ss_len, -1, 3)
                traj_coords.append(add_pseudo_c_beta_from_gly(noising_pos))

        else:
            # noising_quat = rigid.rand_quat([1, ss_len])
            # noising_coord = torch.randn(1, ss_len, 3)* white_noise_scale
            # noising_affine = torch.cat([noising_quat, noising_coord], -1)
            # noising_pos = rigid.affine_to_pos(noising_affine.reshape(-1, 7)).reshape(ss_len, -1, 3)
            # traj_coords.append(add_pseudo_c_beta_from_gly(noising_pos))
            if np.random.rand(1)[0] > loop_mask_p:
                traj_coords.append(add_pseudo_c_beta_from_gly(gt_ss_pos))
            else:
                noising_quat = rigid.rand_quat([1, ss_len])
                noising_coord = torch.randn(1, ss_len, 3) * white_noise_scale
                noising_affine = torch.cat([noising_quat, noising_coord], -1)
                noising_pos = rigid.affine_to_pos(noising_affine.reshape(-1, 7)).reshape(ss_len, -1, 3)
                traj_coords.append(add_pseudo_c_beta_from_gly(noising_pos))

    traj_coords = torch.cat(traj_coords)
    traj_flat12s = get_quataffine(traj_coords)

    return traj_coords, traj_flat12s


def updated_noising_coord(x0, noise_scale):
    noise = torch.randn_like(x0)
    updated_x0 = noise_scale * noise
    return updated_x0


def updated_noising_quat(q0, alpha_t_scale):
    batch_size = q0.shape[0]
    qT = rigid.rand_quat(q0.shape[:-1]).to(q0.device)
    # return qT
    alpha_t = alpha_t_scale * torch.rand((batch_size, ), device=q0.device)
    qt = rigid.slert(q0, qT, alpha_t)
    qt = qt / (qt.square().sum(dim=-1, keepdims=True).sqrt() + 1e-14)
    return qt


def update_rigid_pos(pos, translation, rotation):
    assert len(pos.shape) == 3
    L, N, _ = pos.shape
    # roted_pos = torch.matmul(pos.reshape(-1, 3), torch.transpose(rotation, -1,-2))
    roted_pos = torch.matmul(pos.reshape(-1, 3), rotation)
    updated_pos = roted_pos.reshape(L, N, -1)
    updated_pos = updated_pos + translation[None, None]
    
    return updated_pos


def update_rigid_pos_new(pos, updated_translation, new_rotation):
    assert len(pos.shape) == 3
    L, N, _ = pos.shape
    ca_mass_pos = pos[:, 1].mean(0)
    new_ca_mass_pos = ca_mass_pos + updated_translation
    roted_pos = torch.matmul(pos.reshape(-1, 3) - ca_mass_pos, new_rotation)
    updated_pos = roted_pos.reshape(L, N, -1)
    updated_pos = updated_pos + new_ca_mass_pos[None, None]
    
    return updated_pos


def update_affine(ref_affine, update_affine):
    """Return a new QuatAffine which applies the transformation update first.
    ??? why not QuatAffnie product and translation update
    Args:
      update: Length-7 vector. 3-vector of x, y, and z such that the quaternion
        update is (1, x, y, z) and zero for the 3-vector is the identity
        quaternion. 3-vector for translation concatenated.

    Returns:
      New QuatAffine object.
    """
    quaternion = ref_affine[..., :4]
    translation = ref_affine[..., 4:]
    translation = list(moveaxis(translation, -1, 0))
    rotation = quat_to_rot(quaternion)
    # b, c, d in Supplementary 23
    quaternion_update = update_affine[..., :4] 
    # coordinates
    x = update_affine[..., 4] 
    y = update_affine[..., 5] 
    z = update_affine[..., 6] 
    trans_update = [x, y, z]
    # only use imaginary part to generate new quaternion
    new_quaternion = quat_multiply(quaternion, quaternion_update)
    # new_quaternion = quaternion
    # rotate translation
    trans_update = apply_rot_to_vec(rotation, trans_update)

    new_translation = [
        translation[0] + trans_update[0],
        translation[1] + trans_update[1],
        translation[2] + trans_update[2]]

    return QuatAffine(new_quaternion, new_translation).to_tensor()



def generate_new_affine(num_residues, device):
    quaternion = torch.FloatTensor([1., 0., 0., 0.]).to(device)
    quaternion = quaternion.unsqueeze(0).repeat(num_residues, 1)

    translation = torch.zeros([num_residues, 3]).to(device)
    return QuatAffine(quaternion, translation, unstack_inputs=True)


def gen_coarse_grained_map(config, ca_coord, sstype, train_mode=True):
    adj_dropout = config.coarse_grained.adj_dropout if train_mode else 0.0
    ssedges, ss_adj = make_SS_condition(sstype, ca_coord, adj_dropout)
    return ssedges, ss_adj


def gen_fine_grained_map(config, coord_with_beta, max_len=None):
    # dffferent prob to inpainting task
    seq_len = coord_with_beta.shape[0]
    pair_feature = generate_pair_from_pos(coord_with_beta[None])[0]
    CB_dist_pair = pair_feature[..., 0]
    p_spatial = config.fine_grained.p_spatial
    min_knn = p_spatial[0] # 0.1
    max_knn = p_spatial[1] # 0.5
    knn = int(np.random.uniform(min_knn, max_knn, 1) * seq_len)

    if max_len is not None:
        if seq_len < max_len:
            max_len = seq_len

    central_absidx = torch.randint(0, seq_len if max_len is None else max_len, [1])
    central_knnid = CB_dist_pair[central_absidx.item()]
    knn_idx = torch.argsort(central_knnid)[:knn]
    mask_seq = torch.zeros(seq_len)
    mask_seq = mask_seq.scatter(0, knn_idx, torch.ones_like(knn_idx).float())
    mask_map = mask_seq[:, None] * mask_seq[None]
    masked_pair_map = mask_map[..., None] * pair_feature

    return mask_seq, masked_pair_map


def gen_batch_fine_grained_map(config, coord_with_beta, max_len=None):
    # dffferent prob to inpainting task
    batchsize, seq_len = coord_with_beta.shape[:2]
    pair_feature = generate_pair_from_pos(coord_with_beta)
    CB_dist_pair = pair_feature[..., 0]
    p_spatial = config.fine_grained.p_spatial
    min_knn = p_spatial[0] # 0.1
    max_knn = p_spatial[1] # 0.5
    knn = int(np.random.uniform(min_knn, max_knn, 1) * seq_len)

    if max_len is not None:
        if seq_len < max_len:
            max_len = seq_len
    central_absidx = torch.randint(0, seq_len if max_len is None else max_len, [batchsize])
    batch_central_knnid = torch.stack([CB_dist_pair[bid, central_absidx[bid]] for bid in range(batchsize)])
    knn_idx = torch.argsort(batch_central_knnid)[:, :knn]
    mask_seq = torch.zeros(batchsize, seq_len).to(coord_with_beta.device)
    mask_seq.scatter_(1, knn_idx, torch.ones_like(knn_idx).float())
    mask_map = mask_seq[:, :, None] * mask_seq[:, None]
    masked_pair_map = mask_map[..., None] * pair_feature

    return mask_seq, masked_pair_map

    
def gen_inpainting_mask(config, mask_mode, ca_coord):
    seq_len = ca_coord.shape[0]
    p_rand = config.inpainting.p_rand
    p_lin = config.inpainting.p_lin
    p_spatial = config.inpainting.p_spatial

    min_lin_len = int(p_lin[0] * seq_len) # 0.25
    max_lin_len = int(p_lin[1] * seq_len) # 0.75
    lin_len = torch.randint(min_lin_len, max_lin_len, [1]).item()

    min_knn = p_spatial[0] # 0.1
    max_knn = p_spatial[1] # 0.5
    knn = int(np.random.uniform(min_knn, max_knn, 1) * seq_len)

    if mask_mode == 0: # random
        mask = (torch.rand(1, seq_len) > p_rand).long()

    elif mask_mode == 1: # linear
        start_index = torch.randint(0, seq_len-lin_len, [1])
        mask = torch.ones(1, seq_len)
        mask_idx = start_index[:, None] + torch.arange(lin_len)
        mask.scatter_(1, mask_idx, torch.zeros_like(mask_idx).float())

    elif mask_mode == 2: # spatial
        central_absidx = torch.randint(0, seq_len, [1])
        ca_map = torch.mean(ca_coord[None] - ca_coord[:, None], -1)
        central_knnid = ca_map[central_absidx.item()]
        knn_idx = torch.argsort(central_knnid)[:knn]
        mask = torch.ones(1, seq_len).to(ca_map.device)
        mask.scatter_(1, knn_idx, torch.zeros_like(knn_idx).float())

    return mask.to(ca_coord.device)



def gen_inpainting_mask_batch(config, batchsize, seq_len, mask_mode, gt_pos):
    ca_pos = gt_pos[..., 1, :]
    p_rand = config.inpainting.p_rand
    p_lin = config.inpainting.p_lin
    p_spatial = config.inpainting.p_spatial

    min_lin_len = int(p_lin[0] * seq_len) # 0.25
    max_lin_len = int(p_lin[1] * seq_len) # 0.75
    lin_len = torch.randint(min_lin_len, max_lin_len, [1]).item()

    min_knn = p_spatial[0] # 0.1
    max_knn = p_spatial[1] # 0.5
    knn = int(np.random.uniform(min_knn, max_knn, 1) * seq_len)

    if mask_mode == 0: # random
        mask = (torch.rand(batchsize, seq_len) > p_rand).long()

    elif mask_mode == 1: # linear
        start_index = torch.randint(0, seq_len-lin_len, [batchsize])
        mask = torch.ones(batchsize, seq_len)
        mask_idx = start_index[:, None] + torch.arange(lin_len)
        mask.scatter_(1, mask_idx, torch.zeros_like(mask_idx).float())

    # elif mask_mode == 2: # full
    #     mask = torch.zeros(batchsize, seq_len)

    elif mask_mode == 2: # spatial
        central_absidx = torch.randint(0, seq_len, [batchsize])
        ca_map = torch.mean(ca_pos[:, None] - ca_pos[:, :, None], -1)
        batch_central_knnid = torch.stack([ca_map[bid, central_absidx[bid]] for bid in range(batchsize)])
        knn_idx = torch.argsort(batch_central_knnid)[:, :knn]
        mask = torch.ones(batchsize, seq_len).to(ca_map.device)
        mask.scatter_(1, knn_idx, torch.zeros_like(knn_idx).float())

    return mask.to(ca_pos.device)


def pad_dim(data, dim, max_len):
    """ dim int or [int, int]
    """
    if (isinstance(dim, int) or (isinstance(dim, list) and len(dim) == 0)):
        dim = dim
        if isinstance(dim, int):
            dims = [dim]
        else:
            dims = dim
    else:
        dims = dim
        dim = dim[0]
        
    def convert_pad_shape(pad_shape):
        l = pad_shape[::-1]
        pad_shape = [item for sublist in l for item in sublist]
        return pad_shape

    shape = [d for d in data.shape]
    assert(shape[dim] <= max_len)

    if shape[dim] == max_len:
        return data

    pad_len = max_len - shape[dim]

    pad_shape = []
    for d in dims:
        tmp_pad_shape = [[0, 0]] * d + [[0, pad_len]] + [[0, 0]] * (len(shape) - d -1)
        pad_shape.append(convert_pad_shape(tmp_pad_shape))

    data_pad = F.pad(data, np.sum(pad_shape, 0).tolist(), mode='constant', value=0)
    return data_pad


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return {}
    cat_data = {}
    max_len = max([b['aatype'].shape[0] for b in batch])


    for name in batch[0].keys():
        if name in ['loss_mask', 'len']:
            cat_data[name] = torch.cat([b[name] for b in batch], dim=0)
        elif name in ['pair_res_rel', 'pair_chain_rel']:
            data = torch.cat([pad_dim(b[name], [0, 1], max_len)[None] for b in batch], dim=0)
            cat_data[name] = data
        elif name in ['pdbname']:
            data = [b[name] for b in batch]
            cat_data[name] = data
        else:
            data = torch.cat([pad_dim(b[name], 0, max_len)[None] for b in batch], dim=0)
            cat_data[name] = data

    return cat_data


def data_is_nan(data):
    for k, v in data.items():
        if torch.isnan(v.abs().sum()):
            return True
    return False


def to_tensor(arr):
    if isinstance(arr, np.ndarray):
        if arr.dtype in [np.int64, np.int32]:
            return torch.LongTensor(arr)
        elif arr.dtype in [np.float64, np.float32]:
            return torch.FloatTensor(arr)
        elif arr.dtype == np.bool:
            return torch.BoolTensor(arr)
        else:
            return arr
    else:
        return arr


def pdb_to_data(data_file):
    chain_data = np.load(data_file, allow_pickle=True)
    node_dict = chain_data['node_dict'].item()
    edge_dict = chain_data['edge_dict'].item()

    coord = torch.from_numpy(node_dict['crd']).float()
    aatype = torch.from_numpy(node_dict['AA']).long()
    sstype = torch.from_numpy(node_dict['SS3']).long()
    res_idx = torch.from_numpy(node_dict['seq_index']).long()

    aatype = convert_to_af_aatype(aatype)
    atom_mask = (coord.abs().sum(-1) > 1e-4).float()

    chi_rotamer = all_atom.atom37_to_chi_angles(aatype, coord, atom_mask)
    chi_angles = torch.atan2(chi_rotamer['chi_angles_sin_cos'][..., 1], chi_rotamer['chi_angles_sin_cos'][..., 0])
    chi_angles = torch.nan_to_num(chi_angles, 0.0)
    chi_masks = chi_rotamer['chi_angles_mask']
    chi_angles = chi_angles * chi_masks

    coord_centor = coord[:, 1].mean(0)
    coord = coord - coord_centor[None, None]

    covalant_bond_index = torch.from_numpy(edge_dict['covalant_bond_index']).long()
    covalant_bond_attr = torch.from_numpy(edge_dict['covalant_bond_attr']).long()
    ss_bond_index = torch.from_numpy(edge_dict['ss_bond_index']).long()
    ss_bond_attr = torch.from_numpy(edge_dict['ss_bond_attr']).long()

    if ss_bond_index.shape[1] == 0:
        # fake ss bond
        ss_bond_index = torch.zeros((2,1)).long()
        ss_bond_attr = torch.ones((1,)).long()

    if covalant_bond_index.shape[1] < 16:
        return None

    edge_index = torch.cat([covalant_bond_index, ss_bond_index], axis=-1)
    edge_attr = torch.cat([covalant_bond_attr, ss_bond_attr], dim=-1)

    # data = Data(
    #     aatype=aatype, pos=coord, sstype=sstype, edge_index=edge_index, edge_attr=edge_attr,
    #     n_nodes=torch.LongTensor([coord.shape[0]])
    # )

    data = {
        "aatype": aatype,
        "atom_mask": atom_mask,
        "pos": coord,
        "sstype": sstype,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "res_idx": res_idx,
        "chi_angles": chi_angles,
        "chi_masks": chi_masks,
        "n_nodes": torch.LongTensor([coord.shape[0]])
    }

    return data

def moveaxis(data, source, destination):
  n_dims = len(data.shape)
  dims = [i for i in range(n_dims)]
  if source < 0:
    source += n_dims
  if destination < 0:
    destination += n_dims

  if source < destination:
    dims.pop(source)
    dims.insert(destination, source)
  else:
    dims.pop(source)
    dims.insert(destination, source)

  return data.permute(*dims)


if __name__ == "__main__":
    gen_fine_grained_map(torch.rand(156, 3))