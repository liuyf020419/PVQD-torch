import os
import torch
import logging
import numpy as np
import random
import math
from .dataset import BaseDataset

# from .data_transform import data_to_rigid_groups, perturb_rigid, sequence_mask
from .convert_aatype import convert_to_af_aatype
from alphafold import all_atom
from protein_utils import rigid, backbone
from .data_transform import make_SS_condition, ss_letter2id

logger = logging.getLogger(__name__)


class ProtDiffDataset(BaseDataset):
    def __init__(self, config, data_list, train=True):
        super().__init__()
        self.data_list= data_list
        self.config = config
        self.train_mode = train
        self.load_list(data_list)
        self._epoch = 0


    def load_list(self, data_list):
        self.filelist = []     
        self.mask_range = [] 

        num_samples = self.config.train.num_train_samples if self.train_mode else self.config.train.num_eval_samples
        with open(data_list, 'r') as f:
            for line in f:
                line_split = line.strip().split()
                name = line_split[0]
                graph_size = int(line_split[1])
                self.filelist.append((name, graph_size))

                if len(line_split) == 4:
                    self.mask_range.append((int(line_split[2]), int(line_split[3])))
                
                if len(self.filelist) >= num_samples:
                    break
        logger.info(f'list size: {len(self.filelist)}')

    def __len__(self):
        return len(self.filelist)

    def data_sizes(self):
        return [l[1] for l in self.filelist]
    
    def reset_data(self, epoch):
        self._epoch = epoch

    def __getitem__(self, index:int):
        if index >= len(self.filelist):
            raise IndexError(f'bad index {index}')

        name, _ = self.filelist[index]
        data_file = f'{self.config.data.pdb_data_path}/{name[1:3]}/{name}_graph.npz'
        raw_data = np.load(data_file, allow_pickle=True)

        node_dict = raw_data['node_dict'].item()
        sstype_org = torch.LongTensor(node_dict['SS3'])
        aatype_org = convert_to_af_aatype(torch.LongTensor(node_dict['AA']))
        coord_org = torch.FloatTensor(node_dict['crd'])
        res_idx_org = torch.LongTensor(node_dict['seq_index'])
        res_idx_org = res_idx_org - res_idx_org.min() + 1

        data_len = aatype_org.shape[0]
        crop_size = self.config.data.common.crop_size if self.train_mode else data_len

        ds_rate = self.config.model.score_net.global_config.pair_downsample_rate
        crop_size = int(math.ceil(crop_size / ds_rate) * ds_rate)

        seq_mask = torch.ones((crop_size)).float()

        if data_len > crop_size:
            crop_start = torch.randperm(data_len - crop_size)[0]
            crop_end = crop_start + crop_size
            sstype = sstype_org[crop_start: crop_end]
            aatype = aatype_org[crop_start: crop_end]
            coord = coord_org[crop_start: crop_end]
            res_idx = res_idx_org[crop_start: crop_end]
            res_idx = res_idx - res_idx.min() + 1
        else:
            sstype = torch.zeros((crop_size,), dtype=torch.long) + ss_letter2id['L']
            aatype = torch.zeros((crop_size,), dtype=torch.long) + 20
            coord = torch.zeros((crop_size, 4, 3), dtype=torch.float32)
            res_idx = torch.zeros((crop_size,) , dtype=torch.long)

            sstype[:data_len] = sstype_org
            aatype[:data_len] = aatype_org
            coord[:data_len] = coord_org
            res_idx[:data_len] = res_idx_org
            seq_mask[data_len:] = 0.0

        
        rot, ca_coord = backbone.atom3_to_backbone_frame(aatype, coord[..., :3])
        quat = rigid.rot_to_quat(rot)

        adj_dropout = self.config.data.common.adj_dropout if self.train_mode else 0.0
        ssedges, (ss_adj, beta_parallel) = make_SS_condition(sstype, ca_coord, adj_dropout)

        ss_dropout = self.config.data.common.adj_dropout if self.train_mode else 0.0
        if self.train_mode and ss_dropout > 0:
            ss_mask = torch.ones_like(sstype)
            for ss in ssedges:
                if np.random.uniform() >= ss_dropout:
                    continue
                st, ed = ss[1], ss[2]
                ss_mask[st:ed] = 0
            
            sstype = sstype * ss_mask + (1 - ss_mask) * ss_letter2id['X']

        data = {
            "aatype": aatype,
            "sstype": sstype,
            "res_idx": res_idx,
            "coord": coord,
            "rot": rot,
            "quat": quat,
            "ss_adj": ss_adj.long(),
            "beta_parallel": beta_parallel.long(),
            "seq_mask": seq_mask,
        }

        if self.train_mode:
            return data
        else:
            return name, data


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