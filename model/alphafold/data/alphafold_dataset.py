import os
import torch
import logging
import numpy as np
from .dataset import BaseDataset

from alphafold.data.utils import parsers
from alphafold.data.utils import templates
from alphafold.data.utils import target_features, features

logger = logging.getLogger(__name__)

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
        "template_sum_probs": np.array([[0.0]], dtype=np.float32)
    }
    return template_features

class AlphafoldDataset(BaseDataset):
    '''
    load sample from npz dump file, just for code tuning
    '''
    def __init__(self, config, data_list, train=True):
        # data_list, mmcif_dir, max_template_date, kalign_binary, obsolete_pdbs_path=None):
        super().__init__()
        self.data_list= data_list
        # data_list = config.train.train_list if train else config.train.eval_list
        self.config = config
        #self.config.data.train_mode = train
        self.max_residue_len = config.data.common.max_residue_len
        self.filelist= [fn.strip() for fn in open(data_list)]

        self.mmcif_dir = config.data.mmcif_dir
        self.template_featurizer = templates.HhsearchHitFeaturizer(
            mmcif_dir=self.mmcif_dir,
            max_template_date=config.data.max_template_date,
            max_hits=20,
            kalign_binary_path=config.data.kalign_binary_path,
            release_dates_path=None,
            obsolete_pdbs_path=config.data.obsolete_pdbs_path,
        )
    
    def reset_data(self):
        self.filelist= [fn.strip() for fn in open(self.data_list)]

    def __getitem__(self, index: int) :
        if index >= len(self.filelist):
            raise IndexError(f'bad index {index}')
        # with open(self.filelist[index], 'rb') as f:
        #     sample= np.load(f)
        # sample= {k:torch.from_numpy(v) for k,v in sample.items()}
        # return sample
        prefix = self.filelist[index]
        fasta_file = f'{prefix}.fasta'
        msa_file = f'{prefix}.msa.a3m'
        template_file = f'{prefix}.template_hhr'

        try:

            description, sequence = open(fasta_file).read()[1:].strip().split('\n')
            idx, pdb_code, chain, release_date, resolution, stpos, edpos, clamp_mode = description.split('_')
            stpos, edpos = int(stpos), int(edpos)
            resolution = float(resolution.split('-')[1])

            # prepare templates
            num_templates = min(np.random.randint(1, 20), self.config.data.common.max_templates)
            if not os.path.exists(template_file):
                num_templates = 0
            if num_templates == 0:
                # make fake templates
                templates_features = mk_mock_template(sequence)
                templates_features['template_mask'] = np.array([0], dtype=np.float32)
            else:
                hhsearch_result = open(template_file).read()
                hhsearch_hits = parsers.parse_hhr(hhsearch_result)
                templates_result = self.template_featurizer.get_templates(
                    query_sequence=sequence, hits=hhsearch_hits, 
                    query_pdb_code = pdb_code,
                    query_release_date=None,
                    num_select=num_templates
                )
                templates_features = templates_result.features
                num_templates = templates_features['template_aatype'].shape[0]
                templates_features['template_mask'] = np.array([1]*num_templates, dtype=np.float32)
                if num_templates == 0:
                    # make fake templates
                    templates_features = mk_mock_template(sequence)
                    templates_features['template_mask'] = np.array([0], dtype=np.float32)
            
            msa_line = open(msa_file).read()
            msa = parsers.parse_a3m(msa_line)
            raw_features = {
                **dict(templates_features),
                **target_features.make_structure_features(
                    self.mmcif_dir, pdb_code, chain, stpos, edpos),
                **features.make_msa_features([msa]),
                **features.make_sequence_features(
                        sequence=sequence, description=description, num_res=len(sequence)
                    ),
            }

            raw_features = {k: to_tensor(v) for k, v in raw_features.items()}
            feature_dict = features.process_features(raw_features, self.config)
            feature_dict = target_features.make_target_features(feature_dict, resolution, use_clamped_fape=(clamp_mode == 'clamped-FAPE'))
            if feature_dict['all_atom_mask'].sum()==0:
                logger.warning(f"got sample id {index} {prefix} with no atom info, ignore")
                return None
            return feature_dict
        except Exception as e:
            logger.warning(f'bad sample: {index} {prefix}, exception {e} ')
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

# def to_gpu(x):
#     x = x.contiguous()

#     if torch.cuda.is_available():
#         x = x.cuda(non_blocking=True)
#     return torch.autograd.Variable(x)

# def raw_feature_to_device(raw_features):
#     return {k: to_device(v) for k, v in raw_features.items()}