import torch
import torch.nn.functional as F

import random
import numpy as np
from alphafold.common import residue_constants


AATYPE_NEW_ORDER = torch.LongTensor(residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE)
PERM_MATRIX = torch.zeros((22, 22)).float()
PERM_MATRIX[range(len(residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE)), residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE] = 1.

_MSA_FEATURE_NAMES = [
    'msa', 'deletion_matrix', 'msa_mask', 'msa_row_mask', 'bert_mask',
    'true_msa'
]

restype_atom14_to_atom37 = []  # mapping (restype, atom14) --1> atom37
restype_atom37_to_atom14 = []  # mapping (restype, atom37) --> atom14
restype_atom14_mask = []

for rt in residue_constants.restypes:
    atom_names = residue_constants.restype_name_to_atom14_names[
        residue_constants.restype_1to3[rt]]

    restype_atom14_to_atom37.append([
        (residue_constants.atom_order[name] if name else 0)
        for name in atom_names
    ])

    atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
    restype_atom37_to_atom14.append([
        (atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
        for name in residue_constants.atom_types
    ])

    restype_atom14_mask.append([(1. if name else 0.) for name in atom_names])

# Add dummy mapping for restype 'UNK'
restype_atom14_to_atom37.append([0] * 14)
restype_atom37_to_atom14.append([0] * 37)
restype_atom14_mask.append([0.] * 14)

restype_atom14_to_atom37 = torch.LongTensor(restype_atom14_to_atom37)
restype_atom37_to_atom14 = torch.LongTensor(restype_atom37_to_atom14)
restype_atom14_mask = torch.LongTensor(restype_atom14_mask)

restype_atom37_mask = torch.zeros([21, 37]).float()
for restype, restype_letter in enumerate(residue_constants.restypes):
    restype_name = residue_constants.restype_1to3[restype_letter]
    atom_names = residue_constants.residue_atoms[restype_name]
    for atom_name in atom_names:
        atom_type = residue_constants.atom_order[atom_name]
        restype_atom37_mask[restype, atom_type] = 1


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape

def pad_dim(data, dim, max_len):
    shape = [d for d in data.shape]
    assert(shape[dim] <= max_len)

    if shape[dim] == max_len:
        return data

    pad_len = max_len - shape[dim]
    pad_shape = [[0, 0]] * dim + [[0, pad_len]] + [[0, 0]] * (len(shape) - dim -1)
    data_pad = F.pad(data, convert_pad_shape(pad_shape), mode='constant', value=0)
    return data_pad


def correct_msa_restypes(protein):
    """Correct MSA restype to have the same order as residue_constants."""
    aatype_new_order = AATYPE_NEW_ORDER.to(protein['msa'].device)
    perm_matrix = PERM_MATRIX.to(protein['msa'].device)

    protein['msa'] = aatype_new_order[protein['msa']] #torch.gather(aatype_new_order, protein['msa'], axis=0)
    for k in protein:
        if 'profile' in k:  # Include both hhblits and psiblast profiles
            num_dim = protein[k].shape[-1]
            assert num_dim in [20, 21, 22], (
                'num_dim for %s out of expected range: %s' % (k, num_dim))
            # TODO: tensordot
            import pdb; pdb.set_trace()
            protein[k] = tf.tensordot(protein[k], perm_matrix[:num_dim, :num_dim], 1)
    return protein


def add_distillation_flag(protein, distillation):
    protein['is_distillation'] = torch.from_numpy(np.array(distillation)).float().to(protein['aatype'].device)
    return protein


def squeeze_features(protein):
    """Remove singleton and repeated dimensions in protein features."""
    protein['aatype'] = torch.argmax(protein['aatype'], dim=-1)
    for k in [
        'domain_name', 'msa', 'num_alignments', 'seq_length', 'sequence',
        'superfamily', 'deletion_matrix', 'resolution',
        'between_segment_residues', 'residue_index', 'template_all_atom_masks']:
        if k in protein:
            if protein[k].shape[-1] == 1:
                protein[k] = protein[k].squeeze(-1)

    for k in ['seq_length', 'num_alignments']:
        if k in protein:
            protein[k] = protein[k][0]  # Remove fake sequence dimension
    return protein


def randomly_replace_msa_with_unknown(protein, replace_proportion):
    """Replace a proportion of the MSA with 'X'."""
    msa_mask = (torch.rand(protein['msa'].size()) < replace_proportion).to(protein['aatype'].device)
    x_idx = 20
    gap_idx = 21
    msa_mask = ( msa_mask & (protein['msa'] != gap_idx) )
    protein['msa'] = torch.where(msa_mask, torch.ones_like(protein['msa'])*x_idx, protein['msa'])
    aatype_mask = (torch.rand(protein['aatype'].size()) < replace_proportion).to(protein['aatype'].device)

    protein['aatype'] = torch.where(aatype_mask, torch.ones_like(protein['aatype']) * x_idx, protein['aatype'])
    return protein


def make_seq_mask(protein):
    protein['seq_mask'] = torch.ones(protein['aatype'].size()).float().to(protein['aatype'].device)
    return protein


def make_msa_mask(protein):
    """Mask features are all ones, but will later be zero-padded."""
    protein['msa_mask'] = torch.ones(protein['msa'].size()).float().to(protein['msa'])
    protein['msa_row_mask'] = torch.ones(protein['msa'].size(0)).float().to(protein['msa'])
    return protein


def make_hhblits_profile(protein):
    """Compute the HHblits MSA profile if not already present."""
    if 'hhblits_profile' in protein:
        return protein

    # Compute the profile for every residue (over all MSA sequences).
    protein['hhblits_profile'] = torch.nn.functional.one_hot(protein['msa'], 22).float().mean(0)
    return protein


def make_random_crop_indices(protein, config):
    seq_length = protein['seq_length'].item()
    cfg = config #.train if config.train_mode else config.eval
    crop_size = cfg.common.crop_size
    max_templates = cfg.common.max_templates
    subsample_templates = True if config.train_mode else cfg.common.subsample_templates
    if 'template_mask' in protein:
        num_templates = protein['template_mask'].size(0)
    else:
        num_templates = 0
    num_res_crop_size = min(seq_length, crop_size)

    # Ensures that the cropping of residues and templates happens in the same way
    # across ensembling iterations.
    # Do not use for randomness that should vary in ensembling.
    if subsample_templates:
        # templates_crop_start = np.random.randint(0, num_templates+1)
        templates_crop_start = np.random.randint(0, num_templates - max_templates)
    else:
        templates_crop_start = 0

    num_templates_crop_size = min(num_templates - templates_crop_start, max_templates)
    num_res_crop_start = np.random.randint(0, seq_length - num_res_crop_size + 1)

    templates_select_indices = torch.argsort(torch.rand([num_templates])).to(protein['aatype'].device)
    num_valid_templates = np.random.randint(0, max_templates+1)

    protein['template_mask'] = protein['template_mask'][:max_templates]
    protein['template_mask'][:num_valid_templates] = 0
    
    protein['templates_select_indices'] = templates_select_indices
    protein['num_templates_crop_size'] = num_templates_crop_size
    protein['num_res_crop_start'] = num_res_crop_start
    protein['num_res_crop_size'] = num_res_crop_size
    protein['templates_crop_start'] = templates_crop_start
    # protein['empty_template'] = True if templates_crop_start > num_templates - 1 else False
    # protein['num_valid_templates'] = num_valid_templates

    # if protein['empty_template']:
    #     protein['num_res_crop_start'] = num_templates - 1
    #     protein['num_templates_crop_size'] = 1
    return protein


def fix_templates_aatype(protein):
    """Fixes aatype encoding of templates."""
    # Map one-hot to indices.
    protein['template_aatype'] = torch.argmax(protein['template_aatype'], dim=-1)
  
    aatype_new_order = AATYPE_NEW_ORDER.to(protein['msa'].device)
    protein['template_aatype'] = aatype_new_order[protein['template_aatype']]
    return protein


def make_template_mask(protein):
    protein['template_mask'] = torch.ones(protein['template_domain_names'].shape).float().to(protein['aatype'].device)
    return protein


def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
    """Create pseudo beta features."""
    is_gly = aatype == residue_constants.restype_order['G']
    ca_idx = residue_constants.atom_order['CA']
    cb_idx = residue_constants.atom_order['CB']
    pseudo_beta = torch.where(
        torch.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3]),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :])

    if all_atom_masks is not None:
        pseudo_beta_mask = torch.where(
            is_gly, all_atom_masks[..., ca_idx], all_atom_masks[..., cb_idx]).float()
        return pseudo_beta, pseudo_beta_mask
    else:
        return pseudo_beta


def make_pseudo_beta(protein, prefix=''):
    """Create pseudo-beta (alpha for glycine) position and mask."""
    assert prefix in ['', 'template_']
    protein[prefix + 'pseudo_beta'], protein[prefix + 'pseudo_beta_mask'] = (
        pseudo_beta_fn(
            protein['template_aatype' if prefix else 'all_atom_aatype'],
            protein[prefix + 'all_atom_positions'],
            protein['template_all_atom_masks' if prefix else 'all_atom_mask']))
    return protein


def make_atom14_masks(protein):
    """Construct denser atom positions (14 dimensions instead of 37)."""
    # create the mapping for (residx, atom14) --> atom37, i.e. an array
    # with shape (num_res, 14) containing the atom37 indices for this protein
    if restype_atom14_to_atom37.device != protein['aatype'].device:
        atom14_to_atom37 = restype_atom14_to_atom37.to(protein['aatype'].device)
        atom14_mask = restype_atom14_mask.to(protein['aatype'].device)
        atom37_to_atom14 = restype_atom37_to_atom14.to(protein['aatype'].device)
        atom37_mask = restype_atom37_mask.to(protein['aatype'].device)
    else:
        atom14_to_atom37 = restype_atom14_to_atom37
        atom14_mask = restype_atom14_mask
        atom37_to_atom14 = restype_atom37_to_atom14
        atom37_mask = restype_atom37_mask

    # TODO: gather
    residx_atom14_to_atom37 = atom14_to_atom37[protein['aatype']] 
    residx_atom14_mask = atom14_mask[protein['aatype']] 

    protein['atom14_atom_exists'] = residx_atom14_mask
    protein['residx_atom14_to_atom37'] = residx_atom14_to_atom37

    # create the gather indices for mapping back
    residx_atom37_to_atom14 = atom37_to_atom14[protein['aatype']]
    protein['residx_atom37_to_atom14'] = residx_atom37_to_atom14

    # create the corresponding mask
    residx_atom37_mask = atom37_mask[protein['aatype']] 
    protein['atom37_atom_exists'] = residx_atom37_mask

    return protein


def block_delete_msa(protein, config):
    """Sample MSA by deleting contiguous blocks.

    Jumper et al. (2021) Suppl. Alg. 1 "MSABlockDeletion"

    Arguments:
        protein: batch dict containing the msa
        config: ConfigDict with parameters

    Returns:
        updated protein
    """
    num_seq = protein['msa'].shape[0] 
    block_num_seq = int(num_seq * config.msa_fraction_per_block)
    if config.randomize_num_blocks:
        nb = np.random.randint(1, config.num_blocks + 1)
    else:
        nb = config.num_blocks

    del_block_starts = torch.randint(0, num_seq, [nb]).to(protein['msa'].device) 
    del_blocks = del_block_starts[:, None] + torch.arange(block_num_seq).to(protein['msa'].device)
    del_blocks = torch.clamp(del_blocks, 0, num_seq - 1)
    del_indices = torch.unique(torch.sort(del_blocks.view(-1))[0]) 

    # Make sure we keep the original sequence
    original_indices = torch.arange(num_seq).to(protein['msa'].device)
    original_indices[del_indices] = -1
    original_indices[0] = 0
    keep_indices = torch.where(original_indices >= 0)[0]

    for k in _MSA_FEATURE_NAMES:
        if k in protein:
            protein[k] = protein[k][keep_indices]

    return protein


def sample_msa(protein, max_seq, keep_extra):
    """Sample MSA randomly, remaining sequences are stored as `extra_*`.

    Args:
        protein: batch to sample msa from.
        max_seq: number of sequences to sample.
        keep_extra: When True sequences not sampled are put into fields starting
        with 'extra_*'.

    Returns:
        Protein with sampled msa.
    """
    num_seq = protein['msa'].size(0)
    index = [i for i in range(1, num_seq)]
    random.shuffle(index)
    index = [0] + index
    index_order = torch.LongTensor(index).to(protein['msa'].device)

    num_sel = min(max_seq, num_seq)
    extra_start = min(num_sel, len(index_order)-1)
    sel_seq, not_sel_seq = index_order[:num_sel], index_order[extra_start:]
    # sel_seq, not_sel_seq = index_order[:num_sel], index_order[num_sel:]

    for k in _MSA_FEATURE_NAMES:
        if k in protein:
            if keep_extra:
                protein['extra_' + k] = protein[k][not_sel_seq] 
            protein[k] = protein[k][sel_seq] 

    return protein


def shaped_categorical(probs, epsilon=1e-10):
    ds = probs.size()
    num_classes = ds[-1]
    counts = torch.multinomial(
        (probs + epsilon).view(-1, num_classes), 1
    )
    return torch.reshape(counts, ds[:-1])


def make_masked_msa(protein, config, replace_fraction):
    """Create data for BERT on raw MSA."""
    # Add a random amino acid uniformly
    random_aa = torch.FloatTensor([0.05] * 20 + [0., 0.]).to(protein['msa'].device)
    
    categorical_probs = (
        config.uniform_prob * random_aa +
        config.profile_prob * protein['hhblits_profile'] +
        config.same_prob * torch.nn.functional.one_hot(protein['msa'], 22))

    # Put all remaining probability on [MASK] which is a new column
    pad_shapes = [[0, 0] for _ in range(len(categorical_probs.shape))]
    pad_shapes[-1][1] = 1
    mask_prob = 1. - config.profile_prob - config.same_prob - config.uniform_prob
    assert mask_prob >= 0.
    categorical_probs = torch.nn.functional.pad(categorical_probs, convert_pad_shape(pad_shapes), mode='constant', value=mask_prob)

    sh = protein['msa'].size()
    mask_position = torch.rand(sh).to(protein['msa'].device) < replace_fraction

    bert_msa = shaped_categorical(categorical_probs)
    bert_msa = torch.where(mask_position, bert_msa, protein['msa'])

    # Mix real and masked MSA
    protein['bert_mask'] = mask_position.float()
    protein['true_msa'] = protein['msa']
    protein['msa'] = bert_msa

    return protein

def nearest_neighbor_clusters(protein, gap_agreement_weight=0.):
    """Assign each extra MSA sequence to its nearest neighbor in sampled MSA."""

    # Determine how much weight we assign to each agreement.  In theory, we could
    # use a full blosum matrix here, but right now let's just down-weight gap
    # agreement because it could be spurious.
    # Never put weight on agreeing on BERT mask
    weights = torch.FloatTensor([1]*21 + [gap_agreement_weight] + [1]).to(protein['msa_mask'].device)

    # Make agreement score as weighted Hamming distance
    sample_one_hot = (protein['msa_mask'][:, :, None] *
                        torch.nn.functional.one_hot(protein['msa'], 23))
    extra_one_hot = (protein['extra_msa_mask'][:, :, None] *
                    torch.nn.functional.one_hot(protein['extra_msa'], 23))

    num_seq, num_res, _ = sample_one_hot.size()
    extra_num_seq, _, _ = extra_one_hot.size()

    # Compute tf.einsum('mrc,nrc,c->mn', sample_one_hot, extra_one_hot, weights)
    # in an optimized fashion to avoid possible memory or computation blowup.
    agreement = torch.matmul(
        extra_one_hot.view(extra_num_seq, num_res * 23).float(),
        (sample_one_hot * weights).view(num_seq, num_res * 23).transpose(0, 1))

    # Assign each sequence in the extra sequences to the closest MSA sample
    protein['extra_cluster_assignment'] = torch.argmax(
        agreement, dim=1)

    return protein


def summarize_clusters(protein):
    """Produce profile and deletion_matrix_mean within each cluster."""
    num_seq = protein['msa'].size(0)
    def csum(x):
        x_shape = [i for i in x.shape]
        ret_shape = [num_seq] + x_shape[1:]
        ret = torch.zeros(ret_shape).to(x.device).type_as(x)
        csum_idx = protein['extra_cluster_assignment']
        csum_idx_shape = csum_idx.size()
        expand_shape = [...] + [None] * (len(x_shape) - len(csum_idx_shape))
        repeats = [1] * len(csum_idx_shape) + x_shape[len(csum_idx_shape):]
        csum_idx = csum_idx[expand_shape].repeat(repeats)
        return ret.scatter_add(0, csum_idx, x)

    mask = protein['extra_msa_mask']
    mask_counts = 1e-6 + protein['msa_mask'] + csum(mask)  # Include center

    msa_sum = csum(mask[:, :, None] * torch.nn.functional.one_hot(protein['extra_msa'], 23))
    msa_sum += torch.nn.functional.one_hot(protein['msa'], 23)  # Original sequence
    protein['cluster_profile'] = msa_sum / mask_counts[:, :, None]

    del msa_sum

    del_sum = csum(mask * protein['extra_deletion_matrix'])
    del_sum += protein['deletion_matrix']  # Original sequence
    protein['cluster_deletion_mean'] = del_sum / mask_counts
    del del_sum

    return protein


def crop_extra_msa(protein, max_extra_msa):
    """MSA features are cropped so only `max_extra_msa` sequences are kept."""
    num_seq = protein['extra_msa'].size(0)
    num_sel = min(max_extra_msa, num_seq)

    select_indices = [i for i in range(num_seq)]
    random.shuffle(select_indices)
    select_indices = torch.LongTensor(select_indices[:num_sel]).to(protein['extra_msa'].device)
    for k in _MSA_FEATURE_NAMES:
        if 'extra_' + k in protein:
            protein['extra_' + k] = protein['extra_' + k][select_indices]

    return protein


def delete_extra_msa(protein):
    for k in _MSA_FEATURE_NAMES:
        if 'extra_' + k in protein:
            del protein['extra_' + k]
    return protein


def make_msa_feat(protein):
    """Create and concatenate MSA features."""
    # Whether there is a domain break. Always zero for chains, but keeping
    # for compatibility with domain datasets.
    has_break = protein['between_segment_residues'].float().clamp(0.0, 1.0)
    aatype_1hot = torch.nn.functional.one_hot(protein['aatype'], 21) 

    target_feat = [
        has_break.unsqueeze(-1),
        aatype_1hot, 
    ]

    msa_1hot = torch.nn.functional.one_hot(protein['msa'], 23) 
    has_deletion = protein['deletion_matrix'].clamp(0., 1.)
    deletion_value = torch.atan(protein['deletion_matrix'] / 3.) * (2. / np.pi)

    msa_feat = [
        msa_1hot,
        has_deletion.unsqueeze(-1),
        deletion_value.unsqueeze(-1),
    ]

    if 'cluster_profile' in protein:
        deletion_mean_value = (
            torch.atan(protein['cluster_deletion_mean'] / 3.) * (2. / np.pi))
        msa_feat.extend([
            protein['cluster_profile'],
            deletion_mean_value.unsqueeze(-1),
        ])

    if 'extra_deletion_matrix' in protein:
        protein['extra_has_deletion'] = protein['extra_deletion_matrix'].clamp(0., 1.)
        protein['extra_deletion_value'] = torch.atan(
            protein['extra_deletion_matrix'] / 3.) * (2. / np.pi)

    protein['msa_feat'] = torch.cat(msa_feat, dim=-1)
    protein['target_feat'] = torch.cat(target_feat, dim=-1)
    return protein


def select_feat(protein, feature_list):
    return {k: v for k, v in protein.items() if k in feature_list}


def random_crop_to_size(protein, crop_size, max_templates, shape_schema, crop_params,
                        subsample_templates=False):
    # """Crop randomly to `crop_size`, or keep as is if shorter than that."""
    templates_select_indices = crop_params['templates_select_indices']
    num_templates_crop_size = crop_params['num_templates_crop_size']
    num_res_crop_start = crop_params['num_res_crop_start']
    num_res_crop_size = crop_params['num_res_crop_size']
    templates_crop_start = crop_params['templates_crop_start']
    # empty_template = crop_params['empty_template']

    NUM_RES = protein['aatype'].size(0)
    for k, v in protein.items():
        if k not in shape_schema or (
            'template' not in k and 'NUM_RES' not in shape_schema[k]):
            continue
        res_dim_idx = shape_schema[k].index('NUM_RES') if 'NUM_RES' in shape_schema[k] else -1

        # randomly permute the templates before cropping them.
        if k.startswith('template') and subsample_templates:
            v = v[templates_select_indices] 

        crop_sizes = []
        crop_starts = []
        for i, (dim_size, dim) in enumerate(zip(shape_schema[k], v.size())):
            is_num_res = (dim_size == NUM_RES) and (i == res_dim_idx)
            if i == 0 and k.startswith('template'):
                crop_size = num_templates_crop_size
                crop_start = templates_crop_start
            else:
                crop_start = num_res_crop_start if is_num_res else 0
                crop_size = (num_res_crop_size if is_num_res else
                            (-1 if dim is None else dim))
            crop_sizes.append(crop_size)
            crop_starts.append(crop_start)
        protein[k] = block_slice(v, crop_starts, crop_sizes)

    protein['seq_length'] = torch.from_numpy(np.array(num_res_crop_size)).long().to(protein['aatype'].device)
    # if empty_template:
    #     protein['template_mask'] *= 0
    return protein


def make_fixed_size(protein, shape_schema, msa_cluster_size, extra_msa_size,
                    num_res, num_templates=0):
    """Guess at the MSA and sequence dimensions to make fixed size."""

    pad_size_map = {
        'NUM_RES': num_res,
        'NUM_MSA_SEQ': msa_cluster_size,
        'NUM_EXTRA_SEQ': extra_msa_size,
        'NUM_TEMPLATES': num_templates,
    }

    for k, v in protein.items():
        # Don't transfer this to the accelerator.
        if k == 'extra_cluster_assignment':
            continue
        shape =  [ss for ss in v.shape] #v.shape.as_list()
        schema = shape_schema[k]
        assert len(shape) == len(schema), (
            f'Rank mismatch between shape and shape schema for {k}: '
            f'{shape} vs {schema}')
        pad_size = [
            pad_size_map.get(s2, None) or s1 for (s1, s2) in zip(shape, schema)
        ]
        padding = [[0, p - v.shape[i]] for i, p in enumerate(pad_size)]
        if padding:
            protein[k] = torch.nn.functional.pad(
                v, convert_pad_shape(padding), mode='constant', value=0)
                # tf.pad(
                # v, padding, name=f'pad_to_fixed_{k}')
            # protein[k].set_shape(pad_size)
    return protein


# ugly
def block_slice(x, st, sizes):
    assert(len(st) == len(sizes))
    ed = [s+l for s, l in zip(st, sizes)]

    if len(st) == 1:
        return x[st[0]:ed[0]]
    elif len(st) == 2:
        return x[st[0]:ed[0], st[1]:ed[1]]
    elif len(st) == 3:
        return x[st[0]:ed[0], st[1]:ed[1], st[2]:ed[2]]
    elif len(st) == 4:
        return x[st[0]:ed[0], st[1]:ed[1], st[2]:ed[2], st[3]:ed[3]]
    elif len(st) == 5:
        return x[st[0]:ed[0], st[1]:ed[1], st[2]:ed[2], st[3]:ed[3], st[4]:ed[4]]
    else:
        raise NotImplementedError    


def crop_templates(protein, max_templates):
    for k, v in protein.items():
        if k.startswith('template_'):
            protein[k] = v[:max_templates]
    return protein
