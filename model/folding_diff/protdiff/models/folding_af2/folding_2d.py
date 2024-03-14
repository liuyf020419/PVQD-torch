import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict

from .layers import *
from . import quat_affine
from . import all_atom
from . import r3
from . import utils
from .protein_geom_utils import *
from . import rigid
from . import conv_block
from . import dense_block


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
        # q_scalar.shape == (B, r, 12 * 16)
        self.q_scalar = Linear(msa_channel, num_head * num_scalar_qk)
        # kv_scalar.shape == (B, r, 12 * 16 * 16)
        self.kv_scalar = Linear(msa_channel, num_head * (num_scalar_v + num_scalar_qk))
        # q_point_local.shape == (B, r, 12 * 3 * 4)
        self.q_point_local = Linear(msa_channel, num_head * 3 * num_point_qk)
        # q_point_local.shape == (B, r, 12 * 3 * (4+8))
        self.kv_point_local = Linear(msa_channel, num_head * 3 * (num_point_qk + num_point_v))

        weights = torch.ones((num_head)) * 0.541323855 # np.log(np.exp(1.) - 1.)
        self.trainable_point_weights = nn.Parameter(data=weights, requires_grad=True)

        self.attention_2d = Linear(pair_channel, num_head)

        final_init = 'zeros' if self._zero_initialize_last else 'linear'
        num_final_input = num_head * num_scalar_v + num_head * num_point_v * 4 + num_head * pair_channel
        self.output_projection = Linear(num_final_input, num_output)
        if final_init == 'zeros':
            self.output_projection.weights.data.zero_()

    
    def forward(self, inputs_1d, inputs_2d, affine):
        # num_residues, _ = inputs_1d.shape
        batch_size, num_residues, _ = inputs_1d.shape

        num_head = self.config.num_head
        num_scalar_qk = self.config.num_scalar_qk
        num_point_qk = self.config.num_point_qk
        num_scalar_v = self.config.num_scalar_v
        num_point_v = self.config.num_point_v
        num_output = self.config.num_channel
        dtype= inputs_1d.dtype

        # Construct scalar queries of shape:
        # [num_query_residues, num_head, num_points]
        q_scalar = self.q_scalar(inputs_1d)
        # q_scalar = q_scalar.view(num_residues, num_head, num_scalar_qk)
        q_scalar = q_scalar.view(batch_size, num_residues, num_head, num_scalar_qk)

        # Construct scalar keys/values of shape:
        # [num_target_residues, num_head, num_points]
        kv_scalar = self.kv_scalar(inputs_1d)
        # (r, 12, 16+16)
        # kv_scalar = kv_scalar.view(num_residues, num_head, num_scalar_v + num_scalar_qk)
        kv_scalar = kv_scalar.view(batch_size, num_residues, num_head, num_scalar_v + num_scalar_qk)

        # k_scalar, v_scalar = torch.split(kv_scalar, [num_scalar_qk], dim=-1)
        k_scalar, v_scalar = torch.split(kv_scalar, num_scalar_qk, dim=-1)

        # k_scalar.shape == (r, 12, 16)
        # v_scalar.shape == (r, 12, 16)
        # k_scalar = kv_scalar[..., :num_scalar_qk]
        # v_scalar = kv_scalar[..., num_scalar_qk:]
        
        # Construct query points of shape:
        # [num_residues, num_head, num_point_qk]
    
        # First construct query points in local frame.
        # q_point_local.shape == (r, num_head * 3 * num_point_qk) e.g. (r, 12 * 3 * 4)
        q_point_local = self.q_point_local(inputs_1d)

        # q_point_local = torch.split(q_point_local, 3, dim=-1)
        # q_point_local_dim == 12*4
        q_point_local_dim = q_point_local.size(-1) // 3
        # q_point_local = [[r, 48], [r, 48], [r, 48]]

        # q_point_local = [
        #     q_point_local[..., :q_point_local_dim],
        #     q_point_local[..., q_point_local_dim:2*q_point_local_dim],
        #     q_point_local[..., 2*q_point_local_dim:],
        # ]
        q_point_local = torch.split(q_point_local, q_point_local_dim, dim=-1)

        # affine see generate_new_affine()
        # Project query points into global frame.
        # import pdb; pdb.set_trace()
        # q_point_local[0].shape == (B, N_res, dim//3)
        # affine.to_tensor().shape == (B, N_res, 7)
        # print(q_point_local[0].shape)
        # print(affine.to_tensor().shape)
        # q_point_local [num_residues, num_head, num_point_q]
        # [[r, 12*4], [r, 12*4], [r, 12*4]]
        q_point_global = affine.apply_to_point(q_point_local, extra_dims=1)
        # Reshape query point for later use.
        # q_point_global.shape == [[r, 48], [r, 48], [r, 48]]
        # q_point.shape == [[r, 12, 4], [r, 12, 4], [r, 12, 4]]
        # q_point = [x.view(num_residues, num_head, num_point_qk) for x in q_point_global]
        q_point = [x.view(batch_size, num_residues, num_head, num_point_qk).to(dtype) for x in q_point_global]

        # Construct key and value points.
        # Key points have shape [num_residues, num_head, num_point_qk]
        # Value points have shape [num_residues, num_head, num_point_v]

        # Construct key and value points in local frame.
        # (r, 12 * 3 * (4+8))
        kv_point_local = self.kv_point_local(inputs_1d)
        # kv_point_local = torch.split(kv_point_local, 3, dim=-1)
        kv_point_local_dim = kv_point_local.size(-1) // 3
        # kv_point_local.shape == [(r, 12*(4+8)), (r, 12*(4+8)), (r, 12*(4+8))]
        # kv_point_local = [
        #     kv_point_local[..., :kv_point_local_dim],
        #     kv_point_local[..., kv_point_local_dim:2*kv_point_local_dim],
        #     kv_point_local[..., 2*kv_point_local_dim:],
        # ]
        kv_point_local = torch.split(kv_point_local, kv_point_local_dim, dim=-1)

        # Project key and value points into global frame.
        # [r, 12 * 4, 3] similar like [nres, 4, 3]
        # [[r, 12*4], [r, 12*4], [r, 12*4]]
        kv_point_global = affine.apply_to_point(kv_point_local, extra_dims=1)
        # kv_point_global.shape == [[r, 12*(4+8)], [r, 12*(4+8)], [r, 12*(4+8)]]
        # kv_point_global = [x.view(num_residues, num_head, (num_point_qk + num_point_v)) for x in kv_point_global]
        kv_point_global = [x.view(batch_size, num_residues, num_head, (num_point_qk + num_point_v)).to(dtype) for x in kv_point_global]

        # Split key and value points.
        # v_point.shape == [[r, 12, 4], [r, 12, 4], [r, 12, 4]]
        # k_point.shape == [[r, 12, 4], [r, 12, 4], [r, 12, 4]]
        k_point, v_point = list(
            zip(*[
                # torch.split(x, [num_point_qk,], dim=-1)
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
        # point_weights = point_weights * trainable_point_weights.unsqueeze(1)
        point_weights = point_weights * trainable_point_weights

        # v_point.shape == [[12, r, 4], [12, r, 4], [12, r, 4]]
        # q_point.shape == [[12, r, 4], [12, r, 4], [12, r, 4]]
        # k_point.shape == [[12, r, 4], [12, r, 4], [12, r, 4]]
        v_point = [torch.swapaxes(x, -2, -3) for x in v_point]
        q_point = [torch.swapaxes(x, -2, -3) for x in q_point]
        k_point = [torch.swapaxes(x, -2, -3) for x in k_point]
        # dist2.shape == [(12, r, 1, 4), (12, r, 1, 4), (12, r, 1, 4)]
        # dist2 = [
        #     squared_difference(qx[:, :, None, :], kx[:, None, :, :])
        #     for qx, kx in zip(q_point, k_point)
        # ]
        dist2 = [
            squared_difference(qx[..., None, :], kx[..., None, :, :])
            for qx, kx in zip(q_point, k_point)
        ]
        # sum along the first axis
        # dist2.shape == (12, r, 1, 4)
        dist2 = sum(dist2)
        # attn_qk_point: 
        # ((12, 1, 1, 1)*(12, r, 1, 4)) == sum((12, r, 1, 4), -1) ==> (12, r, 1)
        # attn_qk_point = -0.5 * torch.sum(
        #     point_weights[:, None, None, :] * dist2, dim=-1)
        attn_qk_point = -0.5 * torch.sum(
            point_weights[..., None, None, None] * dist2, dim=-1)
        # (r, 12, 16) ==> (12, r, 16)
        v = torch.swapaxes(v_scalar, -2, -3)
        q = torch.swapaxes(scalar_weights * q_scalar, -2, -3)
        k = torch.swapaxes(k_scalar, -2, -3)
        # attn_qk_scalar.shape == (12, r, r)
        attn_qk_scalar = torch.matmul(q, torch.swapaxes(k, -2, -1))
        # attn_logits.shape == (12, r, 1)
        attn_logits = attn_qk_scalar + attn_qk_point

        # attention_2d.shape == (r, r, 12)
        ch2d_in = inputs_2d.size()[-1]
        attention_2d = self.attention_2d(inputs_2d.reshape(-1, ch2d_in))
        # attention_2d.shape == (12, r, r)
        # attention_2d = torch.permute(attention_2d, [2, 0, 1])
        attention_2d = attention_2d.reshape(batch_size, num_residues, num_residues, -1)
        attention_2d = attention_2d.permute(0, 3, 1, 2)
        attention_2d = attention_2d_weights * attention_2d
        # attn_logits.shape == (12, r, r)
        attn_logits = attn_logits + attention_2d

        # mask_2d = mask * torch.swapaxes(mask, -1, -2)
        # attn_logits = attn_logits - 1e5 * (1. - mask_2d)

        # [num_head, num_query_residues, num_target_residues]
        # attn.shape == (12, r, r)
        attn = F.softmax(attn_logits, dim=-1)

        # [num_head, num_query_residues, num_head * num_scalar_v]
        # (12, r, r)·(12, r, 16) ==> (12, r, 16)
        result_scalar = torch.matmul(attn, v)

        # For point result, implement matmul manually so that it will be a float32
        # on TPU.  This is equivalent to
        # result_point_global = [jnp.einsum('bhqk,bhkc->bhqc', attn, vx)
        #                        for vx in v_point]
        # but on the TPU, doing the multiply and reduce_sum ensures the
        # computation happens in float32 instead of bfloat16.
        # [sum([12, r, r, 4], -2), sum([12, r, r, 4], -2), sum([12, r, r, 4], -2))]
        # [[12, r, 4], [12, r, 4], [12, r, 4]]
        # result_point_global = [torch.sum(
        #     attn[:, :, :, None] * vx[:, None, :, :],
        #     dim=-2) for vx in v_point]
        result_point_global = [torch.sum(
            attn[..., None] * vx[..., None, :, :],
            dim=-2) for vx in v_point]

        # [num_query_residues, num_head, num_head * num_(scalar|point)_v]
        # result_scalar.shape == (r, 12, 16)
        result_scalar = torch.swapaxes(result_scalar, -2, -3)
        # result_point_global [[r, 12, 4], [r, 12, 4], [r, 12, 4]]
        result_point_global = [
            torch.swapaxes(x, -2, -3)
            for x in result_point_global]

        # Features used in the linear output projection. Should have the size
        # [num_query_residues, ?]
        output_features = []
        # result_scalar.shape == (r, 12*16)
        # result_scalar = result_scalar.contiguous().view(num_residues, num_head * num_scalar_v)
        result_scalar = result_scalar.contiguous().view(batch_size, num_residues, num_head * num_scalar_v)
        output_features.append(result_scalar)

        # result_point_global.shape == [[r, 12*8], [r, 12*8], [r, 12*8]]
        # result_point_global = [
        #     r.contiguous().view(num_residues, num_head * num_point_v)
        #     for r in result_point_global]
        result_point_global = [
            r.contiguous().view(batch_size, num_residues, num_head * num_point_v)
            for r in result_point_global]
        # [[r, 12*8], [r, 12*8], [r, 12*8]]
        result_point_local = affine.invert_point(result_point_global, extra_dims=1)
        output_features.extend(result_point_local)

        # output_features.append(torch.sqrt(self._dist_epsilon +
        #                                     torch.square(result_point_local[0]) +
        #                                     torch.square(result_point_local[1]) +
        #                                     torch.square(result_point_local[2])))
        output_features.append(torch.sqrt(self._dist_epsilon +
                                            torch.square(result_point_local[0].float()) +
                                            torch.square(result_point_local[1].float()) +
                                            torch.square(result_point_local[2].float()).to(dtype)))

        # Dimensions: h = heads, i and j = residues,
        # c = inputs_2d channels
        # Contraction happens over the second residue dimension, similarly to how
        # the usual attention is performed.
        # result_attention_over_2d.shape == (r, 12, 16)
        # result_attention_over_2d = torch.einsum('hij, ijc->ihc', attn, inputs_2d)
        result_attention_over_2d = torch.einsum('...hij, ...ijc->...ihc', attn, inputs_2d)

        num_out = num_head * result_attention_over_2d.shape[-1]
        # output_features.shape == (r, 12*16)
        # output_features.append(result_attention_over_2d.view(num_residues, num_out))
        output_features.append(result_attention_over_2d.view(batch_size, num_residues, num_out))

        # final_act.shape == (r, 12*16 + 12*8*3 + 12*8*3 + 12*16)
        final_act = torch.cat(output_features, axis=-1)
        final_act = final_act.to(self.output_projection.weights.dtype)
        # return (r, 384)
        return self.output_projection(final_act)



def generate_pair_from_affine(affine, pair_geom_dict):
    ########################### for affine.quat debug #############################
    pos = rigid.quat_affine_to_pos(affine.quaternion, affine.translation)

    pair_feature = get_map_ch(pos, pair_geom_dict)
    pair_feature = process_pair(pair_feature)
    return pair_feature


def process_pair(pair_feature, mask_dist=20):
        processed_pair_feature = torch.where(pair_feature[0]<mask_dist, pair_feature[0], mask_dist)
        processed_pair_feature[0] = (processed_pair_feature[0] / 10) -1
        processed_pair_feature[1] = processed_pair_feature[1] / math.pi
        processed_pair_feature[2] = processed_pair_feature[2] / math.pi
        processed_pair_feature[3] = (2 * processed_pair_feature[3] / math.pi) - 1

        mask_gms = torch.where(pair_feature[0]>=mask_dist)[0]
        processed_pair_feature[1] = torch.where(mask_gms, 1, processed_pair_feature[1])
        processed_pair_feature[2] = torch.where(mask_gms, 1, processed_pair_feature[2])
        processed_pair_feature[3] = torch.where(mask_gms, 1, processed_pair_feature[3])
        return processed_pair_feature


class FoldBlock(nn.Module):
    def __init__(self, config, global_config, update_affine, msa_channel, pair_channel, conv_pair=True):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.dropout_factor = 0.0 if global_config.deterministic else 1.0

        dropout_rate = 0.0 if global_config.deterministic else config.dropout
        self.dropout = nn.Dropout(p = dropout_rate)

        self.invariant_point_attention = InvariantPointAttention(config, global_config, msa_channel, pair_channel)
        self.attention_layer_norm = nn.LayerNorm(msa_channel)

        final_init = 'zeros' if self.global_config.zero_init else 'linear'

        self.transition = nn.Sequential()
        layers=[]
        in_dim= msa_channel
        for i in range(config.num_layer_in_transition):
            if i < config.num_layer_in_transition -1:
                layers.append(Linear(in_dim, config.num_channel, initializer="relu"))
                layers.append(nn.ReLU())
            else:
                layers.append(Linear(in_dim, config.num_channel, initializer="zeros"))
            in_dim = config.num_channel
        self.transition= nn.Sequential(*layers)

        self.transition_layer_norm = nn.LayerNorm(config.num_channel)

        if update_affine:
            affine_update_size = 6
            self.affine_update = Linear(msa_channel, affine_update_size, initializer=final_init)

        ######################################### TBD ############################################
        pair_channels = config.pair_channels
        channels_idx = np.arange(len(config.pair_channels)-1)
        if conv_pair:
            self.pair_transition = nn.Sequential(
                            conv_block.Resnet_block_noT(in_ch=pair_channels[ch_idx], 
                                                        dropout=config.pair_dropout, 
                                                        out_ch=pair_channels[ch_idx+1]) 
                            for ch_idx in channels_idx)
        else:
            raise ValueError("triangle update has not been implemented")
            

    def forward(self, 
                activations,
                update_affine):

        c = self.config

        affine = activations["affine"]
        pair = generate_pair_from_affine('pair')
        ######################################### TBD ############################################
        act_pair = self.pair_transition(pair)

        affine = quat_affine.QuatAffine.from_tensor(affine) 
        act = activations['act']
        
        act_attn = self.invariant_point_attention(
            inputs_1d=act,
            inputs_2d=act_pair,
            affine=affine)

        act = act + act_attn

        act = self.dropout(act)
        act = self.attention_layer_norm(act)

        act = self.transition(act) + act
        act = self.dropout(act)
        act = self.transition_layer_norm(act)

        if update_affine:
            # This block corresponds to
            # Jumper et al. (2021) Alg. 23 "Backbone update"

            # Affine update
            affine_update = self.affine_update(act)
            affine = affine.pre_compose(affine_update)

        ## frame and postion for output
        outputs = {'affine': affine.to_tensor()}
        affine = affine.apply_rotation_tensor_fn(torch.detach)

        new_activations = {
            'act': act,
            'affine': affine.to_tensor()
        }

        return new_activations, outputs


def generate_new_affine(sequence_mask):
    # num_residues, _ = sequence_mask.shape
    batch_size, num_residues, _ = sequence_mask.shape

    quaternion = torch.FloatTensor([1., 0., 0., 0.]).to(sequence_mask.device)
    # quaternion = quaternion.unsqueeze(0).repeat(num_residues, 1)
    quaternion = quaternion[None, None, :].repeat(batch_size, num_residues, 1)
    ## translation = init_translation[:, None].repeat(1, num_residues, 1) / position_scale

    translation = torch.zeros([batch_size, num_residues, 3]).to(sequence_mask.device)
    return quat_affine.QuatAffine(quaternion, translation, unstack_inputs=True)


def generate_quataffine(quataffine):
    quaternion = quataffine[:, :, :4]
    translation = quataffine[:, :, 4:]
    return quat_affine.QuatAffine(quaternion, translation, unstack_inputs=True)


def l2_normalize(x, dim=-1, epsilon=1e-12):
    # return x / torch.sqrt( torch.sum(x**2, dim=dim, keepdims=True).clamp(min=epsilon) )
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

        self.fold_iterations = nn.ModuleList([FoldBlock(config, global_config, True, msa_channel, pair_channel)
                                                for _ in range(config.num_layer)])

        # self.pair_layer_norm = nn.LayerNorm(pair_channel)

        self.affine_out = Linear(2 * 7, 7, initializer="relu")

    def forward(self, representations):
        c = self.config
        
        act = self.single_layer_norm(representations['single'])
        act = self.initial_projection(act)

        

        affine = generate_quataffine(representations['affine'])

        activations = {'act': act,
                       'affine': affine.to_tensor()
                      }
        outputs = []

        for l_id in range(c.num_layer):
            fold_iterations = self.fold_iterations[l_id]
            activations, output = fold_iterations(
                activations,
                update_affine=True)
            outputs.append(output)

        output = {
            'affine': torch.stack([out['affine'] for out in outputs])
        }

        # Include the activations in the output dict for use by the LDDT-Head.
        output['act'] = activations['act']

        return output

def generate_new_affine(sequence_mask):
    # num_residues, _ = sequence_mask.shape
    batch_size, num_residues, _ = sequence_mask.shape

    quaternion = torch.FloatTensor([1., 0., 0., 0.]).to(sequence_mask.device)
    # quaternion = quaternion.unsqueeze(0).repeat(num_residues, 1)
    quaternion = quaternion[None, None, :].repeat(batch_size, num_residues, 1)
    ## translation = init_translation[:, None].repeat(1, num_residues, 1) / position_scale

    translation = torch.zeros([batch_size, num_residues, 3]).to(sequence_mask.device)
    return quat_affine.QuatAffine(quaternion, translation, unstack_inputs=True)


def generate_quataffine(quataffine):
    quaternion = quataffine[:, :, :4]
    translation = quataffine[:, :, 4:]
    return quat_affine.QuatAffine(quaternion, translation, unstack_inputs=True)


def l2_normalize(x, dim=-1, epsilon=1e-12):
    # return x / torch.sqrt( torch.sum(x**2, dim=dim, keepdims=True).clamp(min=epsilon) )
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

        self.fold_iterations = nn.ModuleList([FoldBlock(config, global_config, True, msa_channel, pair_channel)
                                                for _ in range(config.num_layer)])

        # self.pair_layer_norm = nn.LayerNorm(pair_channel)

        self.affine_out = Linear(2 * 7, 7, initializer="relu")

    def forward(self, representations):
        c = self.config
        
        act = self.single_layer_norm(representations['single'])
        act = self.initial_projection(act)

        

        affine = generate_quataffine(representations['affine'])

        activations = {'act': act,
                       'affine': affine.to_tensor()
                      }
        outputs = []

        for l_id in range(c.num_layer):
            fold_iterations = self.fold_iterations[l_id]
            activations, output = fold_iterations(
                activations,
                update_affine=True)
            outputs.append(output)

        output = {
            'affine': torch.stack([out['affine'] for out in outputs])
        }

        # Include the activations in the output dict for use by the LDDT-Head.
        output['act'] = activations['act']

        return output



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
                                                [c.position_scale] * 3).to(output['act'].device)

        ret['sidechains'] = output['sc']

        # import pdb; pdb.set_trace()
        atom14_pred_positions = r3.vecs_to_tensor(output['sc']['atom_pos'])[-1]
        ret['final_atom14_positions'] = atom14_pred_positions  # (N, 14, 3)
        ret['final_atom14_mask'] = batch['atom14_atom_exists']  # (N, 14)

        atom37_pred_positions = all_atom.atom14_to_atom37(atom14_pred_positions,
                                                        batch)
        atom37_pred_positions = atom37_pred_positions * batch['atom37_atom_exists'][:, :, None]
        ret['final_atom_positions'] = atom37_pred_positions  # (N, 37, 3)

        ret['final_atom_mask'] = batch['atom37_atom_exists']  # (N, 37)
        ret['final_affines'] = ret['traj'][-1]

        if self.compute_loss:
            return ret
        else:
            no_loss_features = ['final_atom_positions', 'final_atom_mask',
                                'representations']
            no_loss_ret = {k: ret[k] for k in no_loss_features}
            return no_loss_ret

