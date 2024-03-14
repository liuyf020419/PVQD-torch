import copy
from typing import List, Mapping, Tuple

import ml_collections
import numpy as np

import torch
import tree

from alphafold.data.utils import data_transforms, msa_identifiers
from alphafold.common import residue_constants

FEATURES = [
    #### Static features of a protein sequence ####
    "aatype",
    "between_segment_residues",
    "deletion_matrix",
    "domain_name",
    "msa",
    "num_alignments",
    "residue_index",
    "seq_length",
    "sequence",
    "all_atom_positions",
    "all_atom_mask",
    "resolution",
    "template_domain_names",
    "template_sum_probs",
    "template_aatype",
    "template_mask",
    "template_all_atom_positions",
    "template_all_atom_masks",
]

template_crop_params = [
    'templates_select_indices',
    'num_templates_crop_size',
    'num_res_crop_start',   
    'num_res_crop_size',
    'templates_crop_start',
    'empty_template'
]


def make_data_config(
      config: ml_collections.ConfigDict,
      num_res: int,
      ) -> Tuple[ml_collections.ConfigDict, List[str]]:
    """Makes a data config for the input pipeline."""
    cfg = copy.deepcopy(config.data)

    feature_names = cfg.common.unsupervised_features
    if cfg.common.use_templates:
        feature_names += cfg.common.template_features

    with cfg.unlocked():
        # cfg_ = cfg.train if cfg.train_mode else cfg.eval
        # cfg_.crop_size = num_res
        cfg.common.crop_size = num_res

    return cfg, feature_names


def process_features(raw_features,
                     config,
                     random_seed = 0):
    """Preprocesses NumPy feature dict using TF pipeline."""
    raw_features = dict(raw_features)
    num_res = int(raw_features['seq_length'][0])
    cfg, feature_names = make_data_config(config, num_res=num_res)

    if 'deletion_matrix_int' in raw_features:
        raw_features['deletion_matrix'] = (
            raw_features.pop('deletion_matrix_int').float())

    tensor_dict = {k: v for k, v in raw_features.items()
                  if k in FEATURES}
    with torch.no_grad():
        features = process_tensors_from_config(tensor_dict, cfg)
    return features


def process_tensors_from_config(tensors, data_config):
    """Apply filters and maps to an existing dataset, based on the config."""
    cfg = data_config #.train if data_config.train_mode else data_config.eval
    tensors = nonensembled_features(tensors, data_config) 
    
    tensors_0 = ensembled_features(tensors.copy(), data_config)
    num_ensemble = 1 if data_config.train_mode else cfg.common.num_ensemble
    num_recycle = data_config.common.num_recycle #np.random.randint(0, data_config.common.num_recycle+1) if data_config.train_mode else data_config.common.num_recycle
    if data_config.train_mode:
        tensors['num_train_recycle'] = torch.LongTensor(num_recycle)
    if data_config.common.resample_msa_in_recycling:
        # Separate batch per ensembling & recycling step. 
        num_ensemble *= num_recycle + 1

    if num_ensemble > 1:
        tensors_list = [tensors_0]
        for i in range(1, num_ensemble):
            tensors_list.append(
                ensembled_features(tensors.copy(), data_config)
            )
        
        tensors = stack_essemble_data(tensors_list)
    else:
        tensors = tree.map_structure(lambda x: x[None],
                                    tensors_0)
    tensors['template_sum_probs'] = tensors['template_sum_probs'].squeeze(-1)
    # tensors['num_recycle'] = num_recycle
    return tensors


def stack_essemble_data(tensors_list):
    tensors = {}
    for k in tensors_list[0].keys():
        tensors[k] = stack_data([t[k] for t in tensors_list])
    return tensors


def stack_data(data_list):
    if len(data_list[0].shape) == 0:
        return torch.stack(data_list)

    shape_0s = [d.shape[0] for d in data_list]
    max_len = max(shape_0s)
    if min(shape_0s) == max_len:
        return torch.stack(data_list)
    else:
        list_to_stack = []
        for d in data_list:
            pad_size = [[0, max_len - d.shape[0]]] + [[0, 0]] * (len(d.shape) - 1)
            pad_size = data_transforms.convert_pad_shape(pad_size)
            list_to_stack.append(
                torch.nn.functional.pad(d, pad_size, value=0)
            )
        return torch.stack(list_to_stack)


def nonensembled_features(tensors, data_config):
    """Input pipeline functions which are not ensembled."""
    common_cfg = data_config.common
    tensors = data_transforms.correct_msa_restypes(tensors)
    tensors = data_transforms.add_distillation_flag(tensors, False)
    # tensors = data_transforms.cast_64bit_ints(tensors) # long tensor
    tensors = data_transforms.squeeze_features(tensors)
    tensors = data_transforms.randomly_replace_msa_with_unknown(tensors, 0.0)  # todo
    tensors = data_transforms.make_seq_mask(tensors)
    tensors = data_transforms.make_msa_mask(tensors)
    tensors = data_transforms.make_hhblits_profile(tensors)
    # tensors = data_transforms.make_random_crop_to_size_seed(tensors)

    if common_cfg.use_templates:
        tensors = data_transforms.fix_templates_aatype(tensors)
        # tensors = data_transforms.make_template_mask(tensors)
        tensors = data_transforms.make_pseudo_beta(tensors, 'template_')
    tensors = data_transforms.make_atom14_masks(tensors)
    # tensors = data_transforms.make_random_crop_indices(tensors, data_config)

    return tensors


def ensembled_features(tensors, data_config):
    """Input pipeline functions that can be ensembled and averaged."""
    common_cfg = data_config.common
    cfg = data_config #.train if data_config.train_mode else data_config.eval
    # eval_cfg = data_config.eval
    if common_cfg.reduce_msa_clusters_by_max_templates:
        pad_msa_clusters = common_cfg.max_msa_clusters - common_cfg.max_templates
    else:
        pad_msa_clusters = common_cfg.max_msa_clusters

    max_msa_clusters = pad_msa_clusters
    max_extra_msa = common_cfg.max_extra_msa
    if data_config.train_mode:
        tensors = data_transforms.block_delete_msa(tensors, common_cfg.block_delete_msa)
    tensors = data_transforms.sample_msa(tensors, max_msa_clusters, keep_extra=True)
    if 'masked_msa' in common_cfg:
        tensors = data_transforms.make_masked_msa(tensors, common_cfg.masked_msa, common_cfg.masked_msa_replace_fraction)
    if common_cfg.msa_cluster_features:
        tensors = data_transforms.nearest_neighbor_clusters(tensors)
        tensors = data_transforms.summarize_clusters(tensors)
    if max_extra_msa:
        tensors = data_transforms.crop_extra_msa(tensors, max_extra_msa)
    else:
        tensors = data_transforms.delete_extra_msa(tensors)
    tensors = data_transforms.make_msa_feat(tensors)

    # if cfg.fixed_size: # and not data_config.train:
    crop_feats = dict(cfg.feat)
    if data_config.train_mode:
        crop_params = {k: v for k, v in tensors.items() if k in template_crop_params}
        tensors = data_transforms.select_feat(tensors, list(crop_feats))
        # tensors = data_transforms.random_crop_to_size(
        #     tensors, 
        #     common_cfg.crop_size,
        #     common_cfg.max_templates,
        #     crop_feats,
        #     crop_params,
        #     common_cfg.subsample_templates
        # )
        if data_config.common.fixed_size:
            tensors = data_transforms.make_fixed_size(
                tensors,
                crop_feats,
                pad_msa_clusters,
                common_cfg.max_extra_msa,
                common_cfg.max_residue_len,
                common_cfg.max_templates
            )
    else:
        tensors = data_transforms.select_feat(tensors, list(crop_feats))
       
        tensors = data_transforms.crop_templates(tensors, common_cfg.max_templates)

    return tensors


def make_sequence_features(
        sequence, description, num_res):
    """Constructs a feature dict of sequence features."""
    features = {}
    features['aatype'] = residue_constants.sequence_to_onehot(
        sequence=sequence,
        mapping=residue_constants.restype_order_with_x,
        map_unknown_to_x=True).numpy()
    features['between_segment_residues'] = np.zeros((num_res,), dtype=np.int32)
    features['domain_name'] = np.array([description.encode('utf-8')],
                                        dtype=np.object_)
    features['residue_index'] = np.array(range(num_res), dtype=np.int32)
    features['seq_length'] = np.array([num_res] * num_res, dtype=np.int32)
    features['sequence'] = np.array([sequence.encode('utf-8')], dtype=np.object_)
    return features


def make_msa_features(msas, slice_start = 0, slice_end = 99999):
    """Constructs a feature dict of MSA features."""
    if not msas:
        raise ValueError('At least one MSA must be provided.')

    int_msa = []
    deletion_matrix = []
    uniprot_accession_ids = []
    species_ids = []
    seen_sequences = set()
    for msa_index, msa in enumerate(msas):
        if not msa:
            raise ValueError(f'MSA {msa_index} must contain at least one sequence.')
        for sequence_index, sequence in enumerate(msa.sequences):
            sequence = sequence[slice_start: slice_end]
            if sequence in seen_sequences:
                continue
            seen_sequences.add(sequence)
            int_msa.append(
                [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence])
            deletion_matrix.append(msa.deletion_matrix[sequence_index][slice_start: slice_end])
            identifiers = msa_identifiers.get_identifiers(
                msa.descriptions[sequence_index])
            uniprot_accession_ids.append(
                identifiers.uniprot_accession_id.encode('utf-8'))
            species_ids.append(identifiers.species_id.encode('utf-8'))

    num_res = len(msas[0].sequences[0])
    num_alignments = len(int_msa)
    features = {}
    features['deletion_matrix_int'] = np.array(deletion_matrix, dtype=np.int32)
    features['msa'] = np.array(int_msa, dtype=np.int32)
    features['num_alignments'] = np.array(
        [num_alignments] * num_res, dtype=np.int32)
    features['msa_uniprot_accession_identifiers'] = np.array(
        uniprot_accession_ids, dtype=np.object_)
    features['msa_species_identifiers'] = np.array(species_ids, dtype=np.object_)
    return features