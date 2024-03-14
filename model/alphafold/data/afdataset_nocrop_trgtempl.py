import os
import torch
import logging
import numpy as np
import random
from .dataset import BaseDataset

from alphafold.data.utils import parsers
from alphafold.data.utils import templates
from alphafold.data.utils import target_features, features

logger = logging.getLogger(__name__)


def load_npy_pdbstructure(pdb_code):
    pdb_npy_dir = '/ps2/hyperbrain/lhchen/data/alphafold2/processed_data/pdb_chain_positions_split'
    pdb_file = f'{pdb_npy_dir}/{pdb_code[1:3]}/{pdb_code}.npy'
    data = np.load(pdb_file, allow_pickle=True).item()
    return {
        "sequence": data['sequence'],
        "all_atom_positions": data['all_atom_positions'],
        "all_atom_mask": data['all_atom_masks']
    }


def mk_mock_template(
    query_sequence, num_temp: int = 1
):
    ln = (
        len(query_sequence)
        if isinstance(query_sequence, str)
        else sum(len(s) for s in query_sequence)
    )
    output_templates_sequence = "A" * ln
    output_confidence_scores = np.full(ln, 1.0)
    templates_all_atom_positions = np.zeros(
        (ln, templates.residue_constants.atom_type_num, 3)
    )
    templates_all_atom_masks = np.zeros((ln, templates.residue_constants.atom_type_num))
    templates_aatype = templates.residue_constants.sequence_to_onehot(
        output_templates_sequence, templates.residue_constants.HHBLITS_AA_TO_ID
    )
    template_features = {
        "template_all_atom_positions": np.tile(
            templates_all_atom_positions[None], [num_temp, 1, 1, 1]
        ),
        "template_all_atom_masks": np.tile(
            templates_all_atom_masks[None], [num_temp, 1, 1]
        ),
        "template_sequence": np.array([f"none".encode()] * num_temp, dtype=object),
        "template_aatype": np.tile(np.array(templates_aatype)[None], [num_temp, 1, 1]),
        "template_confidence_scores": np.tile(
            output_confidence_scores[None], [num_temp, 1]
        ),
        "template_domain_names": np.array([f"none".encode()] * num_temp, dtype=object),
        # "template_release_date": np.array([f"none".encode()] * num_temp, dtype=object),
        "template_sum_probs": np.array([[200.0]], dtype=np.float32)
    }
    return template_features


def make_target_template(
    target, num_temp = 1
):
    sequence = target['sequence']
    atom37 = target['all_atom_positions']
    mask = target['all_atom_mask']
    ln = len(target['sequence'])

    output_templates_sequence = sequence
    output_confidence_scores = np.full(ln, 1.0)
    templates_all_atom_positions = atom37
    templates_all_atom_masks = mask.astype(np.float32)
    templates_aatype = templates.residue_constants.sequence_to_onehot(
        output_templates_sequence, templates.residue_constants.HHBLITS_AA_TO_ID
    )
    template_features = {
        "template_all_atom_positions": np.tile(
            templates_all_atom_positions[None], [num_temp, 1, 1, 1]
        ),
        "template_all_atom_masks": np.tile(
            templates_all_atom_masks[None], [num_temp, 1, 1]
        ),
        "template_sequence": np.array([f"none".encode()] * num_temp, dtype=object),
        "template_aatype": np.tile(np.array(templates_aatype)[None], [num_temp, 1, 1]),
        "template_confidence_scores": np.tile(
            output_confidence_scores[None], [num_temp, 1]
        ),
        "template_domain_names": np.array([f"none".encode()] * num_temp, dtype=object),
        # "template_release_date": np.array([f"none".encode()] * num_temp, dtype=object),
        "template_sum_probs": np.array([[1.0]], dtype=np.float32)
    }
    return template_features


class AFDatasetNoCropTrgTempl(BaseDataset):
    def __init__(self, config, data_list, train=True, length_range=[0, 512], gen_dir = None):
        super().__init__()
        self.data_list= data_list
        self.config = config
        self.train_mode = train
        self.length_range = length_range
        self.gen_dir = gen_dir

        self.mmcif_dir = config.data.mmcif_dir
        self.load_list(data_list)
        # assert(self.config.data.max_template_date <= self.config.train.max_train_date)


    def load_list(self, data_list):
        cfg = self.config.train
        self.filelist = []
        sample_id = 0

        # if self.train_mode:
        #     num_samples = getattr(cfg, "num_train_samples", 10000000)
        # else:
        #     num_samples = getattr(cfg, "num_eval_samples", 100)
        num_samples = 1000000
        with open(data_list) as f:
            for line in f:
                if line.startswith('PDB_Chain'):
                    continue
                
                pdb_chain, map_chain, seq_len, release_date, resolution, clust_size, msa_size = line.strip().split(', ')
                seq_len = int(seq_len)

                if seq_len < self.length_range[0] or seq_len > self.length_range[1]:
                    continue

                resolution = float(resolution)
                clust_size = int(clust_size)
                msa_size = int(msa_size)
                # start = int(start)
                # end = int(end)
                # clamp_fape_mode = clamp == 'True'

                # if self.train_mode and release_date > cfg.max_train_date:
                #     continue
                # if seq_len > cfg.max_seq_len or seq_len < cfg.min_seq_len:
                #     continue
                # if resolution > cfg.max_resolution or resolution < cfg.min_resolution:
                #     continue
                # if msa_size < cfg.min_msa_size:
                #     continue
                
                self.filelist.append(
                    (pdb_chain, map_chain, seq_len, resolution, sample_id)
                )
                sample_id += 1

                if sample_id >= num_samples:
                    break
        # random.shuffle(self.filelist)
        # self.filelist.sort(key = lambda k:k[2])


    def load_target_feature(self, chain_info):
        pdb_chain, map_chain, seq_len, resolution, sample_id = chain_info
        pdb_code, chain = pdb_chain.split('_') 
        
        try:
            # targets = target_features.make_structure_features(
            #             self.mmcif_dir, pdb_code, chain)
            targets = load_npy_pdbstructure(pdb_chain)
            return targets
        except:
            return None

    def load_msa_template(self, chain_name, target):
        msa_dir = f'{self.config.train.msa_dir}/{chain_name}'
        uniref_a3m_file = f'{msa_dir}/uniref.a3m'
        env_a3m_file = f'{msa_dir}/bfd.mgnify30.metaeuk30.smag30.a3m'
        template_feature_file = f'{msa_dir}/template_feature_{self.config.data.max_template_date}.npy'

        msa_line = open(uniref_a3m_file).read()
        num_msas = len(msa_line.split('\n')) // 2
        if os.path.exists(env_a3m_file) and num_msas < 2048:
            msa_line += open(env_a3m_file).read()

        num_msas = len(msa_line.split('\n')) // 2
        if num_msas > 2048:
            msa_line = '\n'.join(msa_line.split('\n')[:4096]) + '\n'
        
        msa = parsers.parse_a3m(msa_line)
        sequence = msa.sequences[0]

        # num_templates = 1
        templates_features = make_target_template(target)
        templates_features['template_mask'] = np.array([1.0], dtype=np.float32)

        # num_templates = min(np.random.randint(1, 20), self.config.data.common.max_templates) 
        # if not os.path.exists(template_feature_file):
        #     num_templates = 0

        # if num_templates == 0:
        #     templates_features = mk_mock_template(sequence)
        #     templates_features['template_mask'] = np.array([0], dtype=np.float32)
        # else:
            # templates_features = np.load(template_feature_file, allow_pickle=True).item()
            # num_valid_templates = templates_features['template_domain_names'].shape[0]

            # num_templates = min(num_templates, num_valid_templates)
            # select_index = [i for i in range(num_valid_templates)]
            # random.shuffle(select_index)
            # select_index = select_index[:num_templates]
            # def _maybe_slice_(k, v, start, end):
            #     if k in ['template_domain_names', 'template_sum_probs']:
            #         return v
            #     elif k == 'template_sequence':
            #         return np.asarray([s.decode() for s in v])
            #     else:
            #         return v[:, start: end]
            # templates_features = {k: _maybe_slice_(k, v[select_index], start, end) for k, v in templates_features.items()}
            # templates_features['template_mask'] = np.array([1]*num_templates, dtype=np.float32)

        return msa, templates_features, sequence

    
    def reset_data(self):
        # self.filelist= [fn.strip() for fn in open(self.data_list)]
        # random.shuffle(self.filelist)
        pass

    def __getitem__(self, index: int) :
        if index >= len(self.filelist):
            raise IndexError(f'bad index {index}')

        chain_info = self.filelist[index]
        pdb_chain, map_chain, seq_len, resolution, sample_id = chain_info

        gen_file = f'{self.gen_dir}/{map_chain[1:3]}/{map_chain}.npy'
        if os.path.exists(gen_file):
            return None
        
        targets_data = self.load_target_feature(chain_info)
        if targets_data is None:
            logger.debug(f'skip sample {pdb_chain}')
            return None
        targets = targets_data
        if np.sum(targets['all_atom_mask'])==0:
            logger.warning(f"got sample id {index} {pdb_chain} (sample id {sample_id}) with no atom info, ignore")
            return None
        try:
            msa_templates_features = self.load_msa_template(map_chain, targets)
            msa, templates_features, _ = msa_templates_features
            sequence = targets['sequence']

            raw_features = {
                **dict(templates_features),
                **targets,
                **features.make_msa_features([msa]),
                **features.make_sequence_features(
                        sequence=sequence, description=pdb_chain, num_res=len(sequence)
                    ),
            }

            raw_features = {k: to_tensor(v) for k, v in raw_features.items()}
            feature_dict = features.process_features(raw_features, self.config)
            feature_dict = target_features.make_target_features(feature_dict, resolution, use_clamped_fape=True)

            return (map_chain, feature_dict)
        except:
            logger.warning(f'bad sample: {pdb_chain}')
            return None

    def __len__(self):
        return len(self.filelist)


def to_tensor(arr):
    if arr.dtype in [np.int64, np.int32]:
        return torch.LongTensor(arr)
    elif arr.dtype in [np.float64, np.float32]:
        return torch.FloatTensor(arr)
    elif arr.dtype == np.bool:
        return torch.BoolTensor(arr)
    else:
        return arr
