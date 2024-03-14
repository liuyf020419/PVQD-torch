import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict
from .layers import *
from alphafold import quat_affine,all_atom,r3, utils



def squared_difference(x, y):
  return torch.square(x - y)


class InvariantPointAttention(nn.Module):
    def __init__(self, config, global_config, msa_channel, pair_channel, dist_epsilon=1e-8):
        super().__init__()
        self._dist_epsilon = dist_epsilon
        self._zero_initialize_last = global_config.zero_init
        self.config = config
        self.global_config = global_config

        num_head = config.num_head
        num_scalar_qk = config.num_scalar_qk
        num_point_qk = config.num_point_qk
        num_scalar_v = config.num_scalar_v
        num_point_v = config.num_point_v
        num_output = config.num_channel
        assert num_scalar_qk > 0
        assert num_point_qk > 0
        assert num_point_v > 0

        self.q_scalar = Linear(msa_channel, num_head * num_scalar_qk)
        self.kv_scalar = Linear(msa_channel, num_head * (num_scalar_v + num_scalar_qk))
        self.q_point_local = Linear(msa_channel, num_head * 3 * num_point_qk)
        self.kv_point_local = Linear(msa_channel, num_head * 3 * (num_point_qk + num_point_v))

        weights = torch.ones((num_head)) * 0.541323855 # np.log(np.exp(1.) - 1.)
        self.trainable_point_weights = nn.Parameter(data=weights, requires_grad=True)

        self.attention_2d = Linear(pair_channel, num_head)
        num_final_input = num_head * num_scalar_v + num_head * num_point_v * 4 + num_head * pair_channel
        self.output_projection = Linear(num_final_input, num_output, initializer='final')

    def forward(self, inputs_1d, inputs_2d, mask, affine):
        batch_size, num_residues, _ = inputs_1d.shape

        num_head = self.config.num_head
        num_scalar_qk = self.config.num_scalar_qk
        num_point_qk = self.config.num_point_qk
        num_scalar_v = self.config.num_scalar_v
        num_point_v = self.config.num_point_v
        # num_output = self.config.num_channel
        dtype= inputs_1d.dtype

        # Construct scalar queries of shape:
        # [batch_size, num_query_residues, num_head, num_points]
        q_scalar = self.q_scalar(inputs_1d)
        q_scalar = q_scalar.view(batch_size, num_residues, num_head, num_scalar_qk)

        # Construct scalar keys/values of shape:
        # [num_target_residues, num_head, num_points]
        kv_scalar = self.kv_scalar(inputs_1d)
        kv_scalar = kv_scalar.view(batch_size, num_residues, num_head, num_scalar_v + num_scalar_qk)

        k_scalar, v_scalar = torch.split(kv_scalar, num_scalar_qk, dim=-1)
        
        # Construct query points of shape:
        # [num_residues, num_head, num_point_qk]

        # First construct query points in local frame.
        q_point_local = self.q_point_local(inputs_1d)
        q_point_local_dim = q_point_local.size(-1) // 3
        q_point_local = torch.split(q_point_local, q_point_local_dim, dim=-1)
        
        # Project query points into global frame.
        q_point_global = affine.apply_to_point(q_point_local, extra_dims=1)
        # Reshape query point for later use.
        q_point = [x.view(batch_size, num_residues, num_head, num_point_qk).to(dtype) for x in q_point_global]

        # Construct key and value points.
        # Key points have shape [num_residues, num_head, num_point_qk]
        # Value points have shape [num_residues, num_head, num_point_v]

        # Construct key and value points in local frame.
        kv_point_local = self.kv_point_local(inputs_1d)
        kv_point_local_dim = kv_point_local.size(-1) // 3
        kv_point_local = torch.split(kv_point_local, kv_point_local_dim, dim=-1)
    
        # Project key and value points into global frame.
        kv_point_global = affine.apply_to_point(kv_point_local, extra_dims=1)
        kv_point_global = [x.view(batch_size, num_residues, num_head, (num_point_qk + num_point_v)).to(dtype) for x in kv_point_global]
        
        # Split key and value points.
        k_point, v_point = list(
            zip(*[
                [x[..., :num_point_qk], x[..., num_point_qk:]]
                for x in kv_point_global
            ]))

        # We assume that all queries and keys come iid from N(0, 1) distribution
        # and compute the variances of the attention logits.
        # Each scalar pair (q, k) contributes Var q*k = 1
        scalar_variance = max(num_scalar_qk, 1) * 1.
        # Each point pair (q, k) contributes Var [0.5 ||q||^2 - <q, k>] = 9 / 2
        point_variance = max(num_point_qk, 1) * 9. / 2

        # Allocate equal variance to scalar, point and attention 2d parts so that
        # the sum is 1.

        num_logit_terms = 3
        scalar_weights = math.sqrt(1.0 / (num_logit_terms * scalar_variance))
        point_weights = math.sqrt(1.0 / (num_logit_terms * point_variance))
        attention_2d_weights = math.sqrt(1.0 / (num_logit_terms))

        # Trainable per-head weights for points.
        trainable_point_weights = F.softplus(self.trainable_point_weights)
        point_weights = point_weights * trainable_point_weights

        v_point = [torch.swapaxes(x, -2, -3) for x in v_point]
        q_point = [torch.swapaxes(x, -2, -3) for x in q_point]
        k_point = [torch.swapaxes(x, -2, -3) for x in k_point]
        dist2 = [
            squared_difference(qx[..., None, :], kx[..., None, :, :])
            for qx, kx in zip(q_point, k_point)
        ]
        dist2 = sum(dist2)
        attn_qk_point = -0.5 * torch.sum(
            point_weights[..., None, None, None] * dist2, dim=-1)

        v = torch.swapaxes(v_scalar, -2, -3)
        q = torch.swapaxes(scalar_weights * q_scalar, -2, -3)
        k = torch.swapaxes(k_scalar, -2, -3)
        attn_qk_scalar = torch.matmul(q, torch.swapaxes(k, -2, -1))
        attn_logits = attn_qk_scalar + attn_qk_point

        attention_2d = self.attention_2d(inputs_2d)
        # attention_2d = torch.permute(attention_2d, [2, 0, 1])
        attention_2d = permute_final_dims(attention_2d, (2,0,1))
        attention_2d = attention_2d_weights * attention_2d
        attn_logits = attn_logits + attention_2d

        mask_2d = mask * torch.swapaxes(mask, -1, -2)
       
        attn_logits = attn_logits - FP16_huge * (1. - mask_2d.unsqueeze(1))

        # [num_head, num_query_residues, num_target_residues]
        attn = F.softmax(attn_logits, dim=-1)

        # [num_head, num_query_residues, num_head * num_scalar_v]
        result_scalar = torch.matmul(attn, v)

        # For point result, implement matmul manually so that it will be a float32
        # on TPU.  This is equivalent to
        # result_point_global = [jnp.einsum('bhqk,bhkc->bhqc', attn, vx)
        #                        for vx in v_point]
        # but on the TPU, doing the multiply and reduce_sum ensures the
        # computation happens in float32 instead of bfloat16.
        attn_f = attn.float()
        
        result_point_global = [torch.sum(
            attn_f[..., None] * vx.float()[..., None, :, :],
            dim=-2) for vx in v_point]

        # [num_query_residues, num_head, num_head * num_(scalar|point)_v]
        result_scalar = torch.swapaxes(result_scalar, -2, -3)
        result_point_global = [
            torch.swapaxes(x, -2, -3)
            for x in result_point_global]

        # Features used in the linear output projection. Should have the size
        # [num_query_residues, ?]
        output_features = []
        result_scalar = result_scalar.contiguous().view(batch_size, num_residues, num_head * num_scalar_v)
        output_features.append(result_scalar)

        result_point_global = [
            r.contiguous().view(batch_size, num_residues, num_head * num_point_v)
            for r in result_point_global]
        result_point_local = affine.invert_point(result_point_global, extra_dims=1)
        output_features.extend(result_point_local)

        output_features.append(torch.sqrt(self._dist_epsilon +
                                            torch.square(result_point_local[0].float()) +
                                            torch.square(result_point_local[1].float()) +
                                            torch.square(result_point_local[2].float())))

        # Dimensions: h = heads, i and j = residues,
        # c = inputs_2d channels
        # Contraction happens over the second residue dimension, similarly to how
        # the usual attention is performed.
        result_attention_over_2d = torch.einsum('...hij, ...ijc->...ihc', attn_f, inputs_2d.float())
        num_out = num_head * result_attention_over_2d.shape[-1]
        output_features.append(result_attention_over_2d.view(batch_size, num_residues, num_out))

        final_act = torch.cat(output_features, axis=-1)
        final_act = final_act.to(self.output_projection.weight.dtype)
        return self.output_projection(final_act)


class FoldBlock(nn.Module):
    def __init__(self, config, global_config, update_affine, msa_channel, pair_channel):
        super().__init__()
        self.config = config
        self.global_config = global_config

        dropout_rate = 0.0 if global_config.deterministic else config.dropout
        self.dropout = nn.Dropout(p = dropout_rate)

        self.invariant_point_attention = InvariantPointAttention(config, global_config, msa_channel, pair_channel)
        self.attention_layer_norm = nn.LayerNorm(msa_channel)

        self.transition = nn.Sequential()
        layers=[]
        in_dim= msa_channel
        for i in range(config.num_layer_in_transition):
            if i < config.num_layer_in_transition -1:
                layers.append(Linear(in_dim, config.num_channel, initializer="relu"))
                layers.append(nn.ReLU())
            else:
                layers.append(Linear(in_dim, config.num_channel, initializer="final"))
            in_dim = config.num_channel
        self.transition= nn.Sequential(*layers)

        self.transition_layer_norm = nn.LayerNorm(config.num_channel)
        self.affine_update = None
        if update_affine:
            affine_update_size = 6
            self.affine_update = Linear(msa_channel, affine_update_size, initializer="final")

        self.rigid_sidechain = MultiRigidSidechain(config.sidechain, self.global_config, msa_channel)


    def forward(self, 
                activations,
                sequence_mask,
                initial_act,
                static_feat_2d=None,
                aatype=None):
        c = self.config
        affine = quat_affine.QuatAffine.from_tensor(activations['affine'].float())

        act = activations['act']
        # if self.global_config.use_checkpoint:
        #     attn = checkpoint(self.invariant_point_attention, act, static_feat_2d, sequence_mask, affine)
        # else:
        attn = self.invariant_point_attention(
            inputs_1d=act,
            inputs_2d=static_feat_2d,
            mask=sequence_mask,
            affine=affine)
        act = act + attn

        act = self.dropout(act)
        act = self.attention_layer_norm(act)

        act = self.transition(act) + act
        act = self.dropout(act)
        act = self.transition_layer_norm(act)

        if self.affine_update is not None:
            # This block corresponds to
            # Jumper et al. (2021) Alg. 23 "Backbone update"

            # Affine update
            affine_update = self.affine_update(act)
            affine = affine.pre_compose(affine_update)

        sc = self.rigid_sidechain(affine.scale_translation(c.position_scale), [act, initial_act], aatype)
        outputs = {'affine': affine.to_tensor(), 'sc': sc}
        affine = affine.apply_rotation_tensor_fn(torch.detach)

        new_activations = {
            'act': act,
            'affine': affine.to_tensor()
        }
        return new_activations, outputs


def generate_new_affine(sequence_mask):
    batch_size, _, num_residues, = sequence_mask.shape
    quaternion = torch.FloatTensor([1., 0., 0., 0.]).to(sequence_mask.device)
    quaternion = quaternion[None,None, :].repeat(batch_size, num_residues, 1)

    translation = torch.zeros([batch_size, num_residues, 3]).to(sequence_mask.device)
    return quat_affine.QuatAffine(quaternion, translation, unstack_inputs=True)


def l2_normalize(x, dim=-1, epsilon=1e-12):
    dtype= x.dtype
    ret= x / torch.sqrt( torch.sum(x.float()**2, dim=dim, keepdims=True).clamp(min=epsilon) )
    return ret.to(dtype)


class AffineGenerator(nn.Module):
    def __init__(self, config, global_config, msa_channel, pair_channel):
        super().__init__()
        self.config = config
        self.global_config = global_config

        self.single_layer_norm = nn.LayerNorm(msa_channel)
        self.initial_projection = Linear(msa_channel, config.num_channel)

        self.fold_iteration = FoldBlock(config, global_config, True, msa_channel, pair_channel)
        self.pair_layer_norm = nn.LayerNorm(pair_channel)

    def forward(self, representations, batch):
        c = self.config
        sequence_mask = batch['seq_mask'][:, None]

        act = self.single_layer_norm(representations['single'])

        initial_act = act
        act = self.initial_projection(act)
        
        affine = generate_new_affine(sequence_mask)

        activations = {'act': act,
                       'affine': affine.to_tensor(),
                      }
        
        act_2d = self.pair_layer_norm(representations['pair'])

        outputs = []
        for _ in range(c.num_layer):
            activations, output = self.fold_iteration(
                activations,
                initial_act=initial_act,
                static_feat_2d=act_2d,
                sequence_mask=sequence_mask,
                aatype=batch['aatype'])
            outputs.append(output)

        output = {
            'affine': torch.stack([out['affine'] for out in outputs]),
            'sc': {
                'angles_sin_cos': torch.stack([out['sc']['angles_sin_cos'] for out in outputs]),
                'unnormalized_angles_sin_cos': torch.stack([out['sc']['unnormalized_angles_sin_cos'] for out in outputs]),
                'atom_pos': r3.stack_vecs([out['sc']['atom_pos'] for out in outputs]),
                'frames': r3.stack_rigids([out['sc']['frames'] for out in outputs]),
            }
        }

        # Include the activations in the output dict for use by the LDDT-Head.
        output['act'] = activations['act']

        return output


def atom14_to_atom37_batch(atom14_data,  # (B,N, 14, ...)
                     batch
                    ):  # (B, N, 37, ...)
  """Convert atom14 to atom37 representation."""
  assert 'residx_atom37_to_atom14' in batch
  assert 'atom37_atom_exists' in batch
  def expand_repeat(x:torch.Tensor, n:int):
    repeat_mode= (1,)*x.dim() +(n,)
    return x.unsqueeze(-1).repeat(repeat_mode)


  atom37_data = torch.gather(atom14_data, -2, expand_repeat(batch['residx_atom37_to_atom14'], 3))
  atom37_data = atom37_data* batch['atom37_atom_exists'][..., None].type_as(atom37_data)

  return atom37_data


class StructureModule(nn.Module):
    def __init__(self, config, global_config, msa_channel, pair_channel, compute_loss=True):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.compute_loss = compute_loss

        self.affine_generator = AffineGenerator(config, global_config, msa_channel, pair_channel)

    def forward(self, representations, batch):
        c = self.config
        ret = {}

        output = self.affine_generator(representations, batch)
        ret['representations'] = {'structure_module': output['act']}

        ret['traj'] = output['affine'] * torch.FloatTensor([1.] * 4 +
                                                [c.position_scale] * 3).to(output['act'])

        ret['sidechains'] = output['sc']

        batch_size,num_res=  batch['atom14_atom_exists'].shape[:2]
        atom14_pred_positions = r3.vecs_to_tensor(output['sc']['atom_pos'])[-1]
        atom14_pred_positions = atom14_pred_positions.view((batch_size, num_res) + atom14_pred_positions.shape[1:])
        ret['final_atom14_positions'] = atom14_pred_positions  # (B,N, 14, 3)
        ret['final_atom14_mask'] = batch['atom14_atom_exists']  # (B,N, 14)
        
        atom37_pred_positions = atom14_to_atom37_batch(atom14_pred_positions,
                                                        batch)
    
        ret['final_atom_positions'] = atom37_pred_positions  # (B,N, 37, 3)

        ret['final_atom_mask'] = batch['atom37_atom_exists']  # (B,N, 37)
        ret['final_affines'] = ret['traj'][-1]

        if self.compute_loss:
            return ret
        else:
            no_loss_features = ['final_atom_positions', 'final_atom_mask',
                                'representations']
            no_loss_ret = {k: ret[k] for k in no_loss_features}
            return no_loss_ret

    def loss(self, value, batch):
        ret = {'loss': 0.}
        ret['metrics'] = {}
        
        # If requested, compute in-graph metrics.
        if self.config.compute_in_graph_metrics:
            atom14_pred_positions = value['final_atom14_positions']
            # Compute renaming and violations.
            value.update(compute_renamed_ground_truth(batch, atom14_pred_positions))
            value['violations'] = find_structural_violations_group(
                batch, atom14_pred_positions, self.config)

            # Several violation metrics:
            violation_metrics = compute_violation_metrics(
                batch=batch,
                atom14_pred_positions=atom14_pred_positions,
                violations=value['violations'])
            ret['metrics'].update(violation_metrics)

        backbone_loss(ret, batch, value, self.config)
   
        if 'renamed_atom14_gt_positions' not in value:
            value.update(compute_renamed_ground_truth(
                batch, value['final_atom14_positions']))
        sc_loss = sidechain_loss(batch, value, self.config)

        ret['loss'] = ((1 - self.config.sidechain.weight_frac) * ret['loss'] +
                        self.config.sidechain.weight_frac * sc_loss['loss'])
        ret['sidechain_fape'] = sc_loss['fape']

        supervised_chi_loss(ret, batch, value, self.config)

        if self.config.structural_violation_loss_weight:
            if 'violations' not in value:
                value['violations'] = find_structural_violations_group(
                    batch, value['final_atom14_positions'], self.config)
            structural_violation_loss(ret, batch, value, self.config)

        return ret
    

def compute_renamed_ground_truth(
        batch: Dict[str, torch.Tensor],
        atom14_pred_positions: torch.Tensor,
        ) -> Dict[str, torch.Tensor]:
    """Find optimal renaming of ground truth based on the predicted positions.

    Jumper et al. (2021) Suppl. Alg. 26 "renameSymmetricGroundTruthAtoms"

    This renamed ground truth is then used for all losses,
    such that each loss moves the atoms in the same direction.
    Shape (N).

    Args:
        batch: Dictionary containing:
        * atom14_gt_positions: Ground truth positions.
        * atom14_alt_gt_positions: Ground truth positions with renaming swaps.
        * atom14_atom_is_ambiguous: 1.0 for atoms that are affected by
            renaming swaps.
        * atom14_gt_exists: Mask for which atoms exist in ground truth.
        * atom14_alt_gt_exists: Mask for which atoms exist in ground truth
            after renaming.
        * atom14_atom_exists: Mask for whether each atom is part of the given
            amino acid type.
        atom14_pred_positions: Array of atom positions in global frame with shape
        (N, 14, 3).
    Returns:
        Dictionary containing:
        alt_naming_is_better: Array with 1.0 where alternative swap is better.
        renamed_atom14_gt_positions: Array of optimal ground truth positions
            after renaming swaps are performed.
        renamed_atom14_gt_exists: Mask after renaming swap is performed.
    """
    batch_size, num_res= batch['atom14_gt_positions'].shape[:2]
    alt_naming_is_better = all_atom.find_optimal_renaming(
        atom14_gt_positions=flatten_prev_dims(batch['atom14_gt_positions'], 2),
        atom14_alt_gt_positions=flatten_prev_dims(batch['atom14_alt_gt_positions'], 2),
        atom14_atom_is_ambiguous=flatten_prev_dims(batch['atom14_atom_is_ambiguous'], 2),
        atom14_gt_exists=flatten_prev_dims(batch['atom14_gt_exists'], 2),
        atom14_pred_positions=flatten_prev_dims(atom14_pred_positions, 2),
        atom14_atom_exists=flatten_prev_dims(batch['atom14_atom_exists'], 2))
    alt_naming_is_better = alt_naming_is_better.view(batch_size, num_res)
    renamed_atom14_gt_positions = (
        (1. - alt_naming_is_better[..., None, None])
        * batch['atom14_gt_positions']
        + alt_naming_is_better[..., None, None]
        * batch['atom14_alt_gt_positions'])

    renamed_atom14_gt_mask = (
        (1. - alt_naming_is_better[..., None]) * batch['atom14_gt_exists']
         + alt_naming_is_better[..., None] * batch['atom14_alt_gt_exists'])

    return {
        'alt_naming_is_better': alt_naming_is_better,  # (N)
        'renamed_atom14_gt_positions': renamed_atom14_gt_positions,  # (N, 14, 3)
        'renamed_atom14_gt_exists': renamed_atom14_gt_mask,  # (N, 14)
    }


def backbone_loss(ret, batch, value, config):
    """Backbone FAPE Loss.

    Jumper et al. (2021) Suppl. Alg. 20 "StructureModule" line 17

    Args:
        ret: Dictionary to write outputs into, needs to contain 'loss'.
        batch: Batch, needs to contain 'backbone_affine_tensor',
        'backbone_affine_mask'.
        value: Dictionary containing structure module output, needs to contain
        'traj', a trajectory of rigids.
        config: Configuration of loss, should contain 'fape.clamp_distance' and
        'fape.loss_unit_distance'.
    """
    affine_trajectory = quat_affine.QuatAffine.from_tensor(value['traj'])
    rigid_trajectory = r3.rigids_from_quataffine(affine_trajectory)

    gt_affine = quat_affine.QuatAffine.from_tensor(
        batch['backbone_affine_tensor'])
    gt_rigid = r3.rigids_from_quataffine(gt_affine)
    backbone_mask = batch['backbone_affine_mask']
    
    def one_frame(r, idx):
        return r3.Rigids(
            rot = r3.Rots(
                r.rot.xx[idx], r.rot.xy[idx], r.rot.xz[idx], 
                r.rot.yx[idx], r.rot.yy[idx], r.rot.yz[idx], 
                r.rot.zx[idx], r.rot.zy[idx], r.rot.zz[idx], 
            ),
            trans = r3.Vecs(r.trans.x[idx], r.trans.y[idx], r.trans.z[idx]),
        )
    def flat_rigids(r):
        return r3.Rigids(
            rot = r3.Rots(
                r.rot.xx.contiguous().view(-1).float(), r.rot.xy.contiguous().view(-1).float(), r.rot.xz.contiguous().view(-1).float(), 
                r.rot.yx.contiguous().view(-1).float(), r.rot.yy.contiguous().view(-1).float(), r.rot.yz.contiguous().view(-1).float(), 
                r.rot.zx.contiguous().view(-1).float(), r.rot.zy.contiguous().view(-1).float(), r.rot.zz.contiguous().view(-1).float(), 
            ),
            trans = r3.Vecs(r.trans.x.contiguous().view(-1).float(), r.trans.y.contiguous().view(-1).float(), r.trans.z.contiguous().view(-1).float()),
        )
    num_traj = value['traj'].size(0)
    fape_loss = []
    # Jumper et al. (2021) Suppl. Sec. 1.11.5 "Loss clamping details"
   
    clamp_distance = config.fape.clamp_distance 
    use_unclamp = getattr(config.fape, 'use_unclamp', False)
    if use_unclamp and 'use_clamped_fape' in batch:
        clamp_distance = config.fape.clamp_distance if bool(batch['use_clamped_fape'][0]) else None

    for traj_id in range(num_traj):
        single_rigid = flat_rigids(one_frame(rigid_trajectory, traj_id))
        gt_rigid_flat= flat_rigids(gt_rigid)
        backbone_mask_flat= backbone_mask.contiguous().view(-1).float()
        single_loss = all_atom.frame_aligned_point_error(
            single_rigid, gt_rigid_flat, backbone_mask_flat,
            single_rigid.trans, gt_rigid_flat.trans,
            backbone_mask_flat,
            l1_clamp_distance=clamp_distance,
            length_scale=config.fape.loss_unit_distance,
            epsilon=1e-12,
        )
        fape_loss.append(single_loss)
    fape_loss = torch.stack(fape_loss)

    # if 'use_clamped_fape' in batch:
    #     # Jumper et al. (2021) Suppl. Sec. 1.11.5 "Loss clamping details"
    #     use_clamped_fape = 1. if bool(batch['use_clamped_fape'][0]) else 0.9
    #     fape_loss_unclamped = []
    #     for traj_id in range(num_traj):
    #         single_rigid = flat_rigids(one_frame(rigid_trajectory, traj_id))
    #         gt_rigid_flat= flat_rigids(gt_rigid)
    #         backbone_mask_flat= backbone_mask.contiguous().view(-1)
    #         single_loss = all_atom.frame_aligned_point_error(
    #             single_rigid, gt_rigid_flat,
    #             backbone_mask_flat,
    #             single_rigid.trans,
    #             gt_rigid_flat.trans,
    #             backbone_mask_flat,
    #             l1_clamp_distance=None,
    #             length_scale=config.fape.loss_unit_distance,
    #             epsilon=1e-12,
    #         )
    #         fape_loss_unclamped.append(single_loss)
    #     fape_loss_unclamped = torch.stack(fape_loss_unclamped)

    #     fape_loss = (fape_loss * use_clamped_fape +
    #                 fape_loss_unclamped * (1 - use_clamped_fape))

    ret['fape'] = fape_loss[-1]
    ret['loss'] = ret['loss'] + torch.mean(fape_loss)


class MultiRigidSidechain(nn.Module):
    def __init__(self, config, global_config, msa_channel):
        super().__init__()
        self.config = config
        self.global_config = global_config

        self.input_projection = Linear(msa_channel, self.config.num_channel)
        self.input_projection_1 = Linear(msa_channel, self.config.num_channel)
        self.resblock1 = nn.ModuleList()
        self.resblock2 = nn.ModuleList()
        for i in range(config.num_residual_block):
            self.resblock1.append(
                Linear(self.config.num_channel, self.config.num_channel, initializer='relu')
            )
            self.resblock2.append(
                Linear(self.config.num_channel, self.config.num_channel, initializer='final')
            )
        self.unnormalized_angles = Linear(self.config.num_channel, 14)

    def forward(self, affine:quat_affine.QuatAffine, representations_list, aatype):

        assert(len(representations_list) == 2)
        act = [
            self.input_projection(F.relu(representations_list[0])),
            self.input_projection_1(F.relu(representations_list[1])),
        ]
        act = sum(act)

        # Mapping with some residual blocks.
        for bidx in range(self.config.num_residual_block):
            old_act = act
            act = self.resblock1[bidx](F.relu(act))
            act = self.resblock2[bidx](F.relu(act))
            act = act + old_act
        
        batch_size, num_res = act.shape[:2]
        unnormalized_angles = self.unnormalized_angles(F.relu(act))

        unnormalized_angles = unnormalized_angles.view(batch_size, num_res, 7, 2)
        angles = l2_normalize(unnormalized_angles, dim=-1)

        outputs = {
            'angles_sin_cos': angles,  # jnp.ndarray (N, 7, 2)
            'unnormalized_angles_sin_cos':
                unnormalized_angles,  # jnp.ndarray (N, 7, 2)
        }

        affine_tensor= flatten_prev_dims(affine.to_tensor(),2)
        affine= quat_affine.QuatAffine.from_tensor(affine_tensor)
        aatype=flatten_prev_dims(aatype,2)
        angles= flatten_prev_dims(angles, 2)
        backb_to_global = r3.rigids_from_quataffine(affine)
        all_frames_to_global = all_atom.torsion_angles_to_frames(
            aatype, backb_to_global, angles
        )
        pred_positions = all_atom.frames_and_literature_positions_to_atom14_pos(
            aatype,all_frames_to_global
        )
        outputs.update({
            'atom_pos': pred_positions,  # r3.Vecs (N, 14)
            'frames': all_frames_to_global,  # r3.Rigids (N, 8)
        })
        return outputs

def find_structural_violations_group(
    batch,
    atom14_pred_positions,
    config
):
    assert len(atom14_pred_positions.shape) ==4, f"atom14_pred_positions shape {atom14_pred_positions.shape}"
    batch_size= atom14_pred_positions.shape[0]

    vio_list= []
    for i in range(batch_size):
        a14_pos_curr= atom14_pred_positions[i]
        curr_batch= {}
        for k,v in batch.items():
            assert batch[k].shape[0] == batch_size
            curr_batch[k]= v[i]
        vio_curr= find_structural_violations(curr_batch, a14_pos_curr, config)
        vio_list.append(vio_curr)
    out={}
    for k1 in ('between_residues','within_residues'):
        out[k1]={}
        for k2 in vio_list[0][k1].keys():
            if len(vio_list[0][k1][k2].shape) ==0:
                out[k1][k2]= sum(v[k1][k2] for v in vio_list)/ len(vio_list)
            else:
                out[k1][k2] = torch.stack([v[k1][k2] for v in vio_list],dim=0)
    out['total_per_residue_violations_mask']= torch.stack([v['total_per_residue_violations_mask'] for v in vio_list], dim=0)
    return out


    


def find_structural_violations(
    batch,
    atom14_pred_positions,  # (N, 14, 3)
    config
    ):
    """Computes several checks for structural violations."""
    atom14_pred_positions= atom14_pred_positions.float()
    # Compute between residue backbone violations of bonds and angles.

    connection_violations = all_atom.between_residue_bond_loss(
        pred_atom_positions=atom14_pred_positions,
        pred_atom_mask=batch['atom14_atom_exists'].float(), #.astype(jnp.float32),
        residue_index=batch['residue_index'].float(), #.astype(jnp.float32),
        aatype=batch['aatype'],
        tolerance_factor_soft=config.violation_tolerance_factor,
        tolerance_factor_hard=config.violation_tolerance_factor)

    # Compute the Van der Waals radius for every atom
    # (the first letter of the atom name is the element type).
    # Shape: (N, 14).
    atomtype_radius = torch.FloatTensor([
        residue_constants.van_der_waals_radius[name[0]]
        for name in residue_constants.atom_types
    ]).to(batch['aatype'].device)
    num_res = batch['aatype'].size(0)
    atom14_atom_radius = batch['atom14_atom_exists'] * torch.gather(
        atomtype_radius.unsqueeze(0).repeat(num_res, 1), 1, batch['residx_atom14_to_atom37'])

    # Compute the between residue clash loss.
    between_residue_clashes = all_atom.between_residue_clash_loss(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=batch['atom14_atom_exists'],
        atom14_atom_radius=atom14_atom_radius,
        residue_index=batch['residue_index'],
        overlap_tolerance_soft=config.clash_overlap_tolerance,
        overlap_tolerance_hard=config.clash_overlap_tolerance)
    
    

    # Compute all within-residue violations (clashes,
    # bond length and angle violations).
    restype_atom14_bounds = residue_constants.make_atom14_dists_bounds(
        overlap_tolerance=config.clash_overlap_tolerance,
        bond_length_tolerance_factor=config.violation_tolerance_factor)
    
    atom14_dists_lower_bound = torch.gather(
        restype_atom14_bounds['lower_bound'].to(batch['aatype'].device), 0, batch['aatype'][..., None, None].repeat(1, 14, 14))
    atom14_dists_upper_bound = torch.gather(
        restype_atom14_bounds['upper_bound'].to(batch['aatype'].device), 0, batch['aatype'][..., None, None].repeat(1, 14, 14))
    within_residue_violations = all_atom.within_residue_violations(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=batch['atom14_atom_exists'],
        atom14_dists_lower_bound=atom14_dists_lower_bound,
        atom14_dists_upper_bound=atom14_dists_upper_bound,
        tighten_bounds_for_loss=0.0)
    # Combine them to a single per-residue violation mask (used later for LDDT).
    per_residue_violations_mask = torch.max(torch.stack([
        connection_violations['per_residue_violation_mask'],
        torch.max(between_residue_clashes['per_atom_clash_mask'], dim=-1)[0],
        torch.max(within_residue_violations['per_atom_violations'],
                dim=-1)[0]]), dim=0)[0]

    return {
        'between_residues': {
            'bonds_c_n_loss_mean':
                connection_violations['c_n_loss_mean'],  # ()
            'angles_ca_c_n_loss_mean':
                connection_violations['ca_c_n_loss_mean'],  # ()
            'angles_c_n_ca_loss_mean':
                connection_violations['c_n_ca_loss_mean'],  # ()
            'connections_per_residue_loss_sum':
                connection_violations['per_residue_loss_sum'],  # (N)
            'connections_per_residue_violation_mask':
                connection_violations['per_residue_violation_mask'],  # (N)
            'clashes_mean_loss':
                between_residue_clashes['mean_loss'],  # ()
            'clashes_per_atom_loss_sum':
                between_residue_clashes['per_atom_loss_sum'],  # (N, 14)
            'clashes_per_atom_clash_mask':
                between_residue_clashes['per_atom_clash_mask'],  # (N, 14)
        },
        'within_residues': {
            'per_atom_loss_sum':
                within_residue_violations['per_atom_loss_sum'],  # (N, 14)
            'per_atom_violations':
                within_residue_violations['per_atom_violations'],  # (N, 14),
        },
        'total_per_residue_violations_mask':
            per_residue_violations_mask,  # (N)
    }

# def batch_mask_mean(mask, value, eps= 1e-10):
#     dims = list(range(len(value.shape)))[1:]
#     mask= mask.float()
#     value= value.float()
#     broadcast_factor = 1.
#     for axis_ in dims:
#         value_size = value.size(axis_)
#         mask_size = mask.size(axis_)
#         if mask_size == 1:
#             broadcast_factor *= value_size
#         else:
#             assert mask_size == value_size
#     return torch.sum( mask *value, dim=dims) / (torch.sum(mask, dim=dims) * broadcast_factor + eps)

def extreme_ca_ca_distance_violations_group(
    pred_atom_positions, #: jnp.ndarray,  # (B,N, 37(14), 3)
    pred_atom_mask, #: jnp.ndarray,  # (B,N, 37(14))
    residue_index, #: jnp.ndarray,  # (B,N)
    max_angstrom_tolerance=1.5
    ):
    batch_size= pred_atom_mask.shape[0]
    outs=[]
    
    for i in range(batch_size):
        out= all_atom.extreme_ca_ca_distance_violations(
            pred_atom_positions[i], pred_atom_mask[i],residue_index[i],
            max_angstrom_tolerance
        )
        outs.append(out)
    outs= sum(outs)/ len(outs)
    return outs
    

def compute_violation_metrics(
    batch, #: Dict[str, jnp.ndarray],
    atom14_pred_positions, #: jnp.ndarray,  # (N, 14, 3)
    violations, #: Dict[str, jnp.ndarray],
    ): # -> Dict[str, jnp.ndarray]:
  """Compute several metrics to assess the structural violations."""
  ret = {}
  extreme_ca_ca_violations = extreme_ca_ca_distance_violations_group(
      pred_atom_positions=atom14_pred_positions,
      pred_atom_mask=batch['atom14_atom_exists'].float(), #.astype(jnp.float32),
      residue_index=batch['residue_index'].float())
  ret['violations_extreme_ca_ca_distance'] = extreme_ca_ca_violations
  ret['violations_between_residue_bond'] = utils.mask_mean(
      mask=batch['seq_mask'],
      value=violations['between_residues'][
          'connections_per_residue_violation_mask'])
  ret['violations_between_residue_clash'] = utils.mask_mean(
      mask=batch['seq_mask'],
      value=torch.max(
          violations['between_residues']['clashes_per_atom_clash_mask'],
          dim=-1)[0])
  ret['violations_within_residue'] = utils.mask_mean(
      mask=batch['seq_mask'],
      value=torch.max(
          violations['within_residues']['per_atom_violations'], axis=-1)[0])
  ret['violations_per_residue'] = utils.mask_mean(
      mask=batch['seq_mask'],
      value=violations['total_per_residue_violations_mask'])
  return ret


def sidechain_loss(batch, value, config):
  """All Atom FAPE Loss using renamed rigids."""
  # Rename Frames
  # Jumper et al. (2021) Suppl. Alg. 26 "renameSymmetricGroundTruthAtoms" line 7
  alt_naming_is_better = value['alt_naming_is_better']

  renamed_gt_frames = (
      (1. - alt_naming_is_better[..., None, None])
      * batch['rigidgroups_gt_frames']
      + alt_naming_is_better[..., None, None]
      * batch['rigidgroups_alt_gt_frames'])

  flat_gt_frames = r3.rigids_from_tensor_flat12(
      renamed_gt_frames.view(-1, 12))
  flat_frames_mask = batch['rigidgroups_gt_exists'].contiguous().view(-1)

  flat_gt_positions = r3.vecs_from_tensor(value['renamed_atom14_gt_positions'].contiguous().view(-1, 3))
  flat_positions_mask = value['renamed_atom14_gt_exists'].contiguous().view(-1)

  # Compute frame_aligned_point_error score for the final layer.
  pred_frames = value['sidechains']['frames']
  pred_positions = value['sidechains']['atom_pos']

  def _slice_last_layer_and_flatten(x):
      return x[-1].view(-1)
  flat_pred_frames = r3.tree_map_rigids(
      _slice_last_layer_and_flatten, pred_frames)
  flat_pred_positions = r3.tree_map_vecs(
      _slice_last_layer_and_flatten, pred_positions)
  # FAPE Loss on sidechains
  
  fape = all_atom.frame_aligned_point_error(
      pred_frames=flat_pred_frames,
      target_frames=flat_gt_frames,
      frames_mask=flat_frames_mask,
      pred_positions=flat_pred_positions,
      target_positions=flat_gt_positions,
      positions_mask=flat_positions_mask,
      l1_clamp_distance=config.sidechain.atom_clamp_distance,
      length_scale=config.sidechain.length_scale,
      epsilon=1e-4)

  return {
      'fape': fape,
      'loss': fape}



def supervised_chi_loss(ret, batch, value, config):
  """Computes loss for direct chi angle supervision.

  Jumper et al. (2021) Suppl. Alg. 27 "torsionAngleLoss"

  Args:
    ret: Dictionary to write outputs into, needs to contain 'loss'.
    batch: Batch, needs to contain 'seq_mask', 'chi_mask', 'chi_angles'.
    value: Dictionary containing structure module output, needs to contain
      value['sidechains']['angles_sin_cos'] for angles and
      value['sidechains']['unnormalized_angles_sin_cos'] for unnormalized
      angles.
    config: Configuration of loss, should contain 'chi_weight' and
      'angle_norm_weight', 'angle_norm_weight' scales angle norm term,
      'chi_weight' scales torsion term.
  """
  eps = 1e-6
  sequence_mask = batch['seq_mask']
  batch_size, num_res = sequence_mask.shape[:2]
  chi_mask = batch['chi_mask'].float()
  pred_angles = value['sidechains']['angles_sin_cos'].view(-1, batch_size, num_res, 7, 2)
  pred_angles = pred_angles[..., 3:, :]


  residue_type_one_hot = F.one_hot(
      batch['aatype'], residue_constants.restype_num + 1)[None].float().to(batch['aatype'].device)
  chi_pi_periodic = torch.einsum('...ijk, ...kl->...ijl', residue_type_one_hot,
                               torch.FloatTensor(residue_constants.chi_pi_periodic).to(chi_mask.device))

  sin_cos_true_chi = batch['chi_angles']

  # This is -1 if chi is pi-periodic and +1 if it's 2pi-periodic
  shifted_mask = (1 - 2 * chi_pi_periodic)[..., None]
  sin_cos_true_chi_shifted = shifted_mask * sin_cos_true_chi
  
  sq_chi_error = torch.sum(
      squared_difference(sin_cos_true_chi, pred_angles), -1)
  sq_chi_error_shifted = torch.sum(
      squared_difference(sin_cos_true_chi_shifted, pred_angles), -1)
  sq_chi_error = torch.min(torch.stack([sq_chi_error, sq_chi_error_shifted]), dim=0)[0]

  sq_chi_loss = utils.mask_mean(mask=chi_mask[None], value=sq_chi_error)
  ret['chi_loss'] = sq_chi_loss
  ret['loss'] = ret['loss'] + config.chi_weight * sq_chi_loss
  unnormed_angles = value['sidechains']['unnormalized_angles_sin_cos'].view(-1,batch_size, num_res, 7, 2)

  angle_norm = torch.sqrt(torch.sum(torch.square(unnormed_angles), dim=-1) + eps)
  norm_error = torch.abs(angle_norm - 1.)
  angle_norm_loss = utils.mask_mean(mask=sequence_mask[None, ..., None],
                                    value=norm_error)

  ret['angle_norm_loss'] = angle_norm_loss
  ret['loss'] = ret['loss'] + config.angle_norm_weight * angle_norm_loss



def structural_violation_loss(ret, batch, value, config):
  """Computes loss for structural violations."""
  assert config.sidechain.weight_frac

  # Put all violation losses together to one large loss.
  violations = value['violations']
  num_atoms = torch.sum(batch['atom14_atom_exists']).float()
  
  ret['loss'] = ret['loss'] + (config.structural_violation_loss_weight * (
      violations['between_residues']['bonds_c_n_loss_mean'] +
      violations['between_residues']['angles_ca_c_n_loss_mean'] +
      violations['between_residues']['angles_c_n_ca_loss_mean'] +
      torch.sum(
          violations['between_residues']['clashes_per_atom_loss_sum'] +
          violations['within_residues']['per_atom_loss_sum']) /
      (1e-6 + num_atoms)))

