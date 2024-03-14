import torch
from torch import nn
from torch.nn import functional as F
# from torch.utils.checkpoint import checkpoint_sequential, checkpoint
from .utils import checkpoint_function, checkpoint_sequential, MultiArgsSequential, ResModule
from alphafold.layers import *
from alphafold import quat_affine
from alphafold.common import residue_constants

from .layers import *


class TemplatePairLayer(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.config=config
        self.global_config= global_config
        drop_factor = (
            config.triangle_attention_starting_node.dropout_rate
            if not global_config.deterministic
            else 0
        )
        self.dropout_row= DropoutRowwise(drop_factor)
        self.dropout_col = DropoutColumnwise(drop_factor)
        self.triangle_attention_starting_node = TriangleAttention(
            config.triangle_attention_starting_node,
            global_config,
            config.triangle_attention_starting_node.value_dim,
            is_template_stack=True
        )
        self.triangle_attention_ending_node = TriangleAttention(
            config.triangle_attention_ending_node,
            global_config,
            config.triangle_attention_ending_node.value_dim,
            is_template_stack=True
        )
        self.triangle_multiplication_outgoing = TriangleMultiplication(
            config.triangle_multiplication_outgoing,
            global_config,
            config.triangle_attention_ending_node.value_dim,
            is_template_stack=True
        )
        self.triangle_multiplication_incoming=TriangleMultiplication(
            config.triangle_multiplication_incoming,
            global_config,
            config.triangle_attention_ending_node.value_dim,
            is_template_stack=True
        )
        self.pair_transition = Transition(config.pair_transition, global_config, 
            config.triangle_attention_starting_node.value_dim)
    
    def forward(self, pair_act, pair_mask):
        mask_ex = pair_mask.unsqueeze(1)
        pair_act = pair_act + self.dropout_row(
            self.triangle_attention_starting_node(pair_act, mask_ex)
        )
        pair_act = pair_act + self.dropout_col(
            self.triangle_attention_ending_node(pair_act, mask_ex)
        )
        pair_act = pair_act + self.dropout_row(
            self.triangle_multiplication_outgoing(pair_act, mask_ex)
        )
        pair_act = pair_act + self.dropout_row(
            self.triangle_multiplication_incoming(pair_act, mask_ex)
        )
        pair_act = pair_act + self.pair_transition(pair_act, mask_ex)
        # return same as input, so we can nn.Sequential it
        return pair_act, pair_mask

class TemplatePairStack(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.layers= MultiArgsSequential(
            *[TemplatePairLayer(config, global_config) for _ in range(config.num_block)]
        )
        self.cp_segments= config.checkpoint
    
    def forward(self, pair_act,pair_mask, subbatch=0):
        """
            TODO: support subbatch during inference
        """
        
        return checkpoint_sequential(
            self.layers,
            self.cp_segments,
            (pair_act, pair_mask)
        )
        

class TemplateAttention(nn.Module):
    def __init__(self, config, global_config, query_num_channels):
        super().__init__()
        self.config= config
        self.global_config= global_config
        self.attention = Attention(
            config.attention, global_config, 
            (query_num_channels, self.config.template_pair_stack.triangle_attention_ending_node.value_dim), 
            query_num_channels
        )
    
    def forward(self, query_embedding:torch.Tensor,template_pair_representation:torch.Tensor,template_mask:torch.Tensor):
        num_templates = template_mask.size(-1)
        num_channels = (self.config.template_pair_stack
                    .triangle_attention_ending_node.value_dim)
        batch_size, num_res,_, query_num_channels = query_embedding.shape
        # Cross attend from the query to the templates along the residue
        # dimension by flattening everything else into the batch dimension.
        # Jumper et al. (2021) Suppl. Alg. 17 "TemplatePointwiseAttention"
        flat_query = query_embedding.view(batch_size, num_res * num_res, 1, query_num_channels)
        
        flat_templates= permute_final_dims(template_pair_representation, (1,2,0,3))\
            .view(batch_size, num_res*num_res, num_templates, num_channels)

        bias = (FP16_huge* (template_mask[...,None, None, None, :] - 1.))
        embedding = self.attention(
            flat_query, flat_templates, bias
        )
       
        embedding = embedding.view(batch_size, num_res, num_res, query_num_channels)
        tmask = (torch.sum(template_mask.view(batch_size,-1),1) > 0.).to(embedding.dtype)
        tmask = tmask[:, None, None, None]
        embedding = embedding * tmask
        return embedding



class TemplateEmbedding(nn.Module):
    def __init__(self, config, global_config, query_num_channels):
        super().__init__()
        self.config = config
        self.global_config = global_config 
        num_channels = (self.config.template_pair_stack
                        .triangle_attention_ending_node.value_dim)

        # 88 = 39 + 1 + 22 + 22 + 1 + 1 + 1 + 1
        input_dim = config.template_dgrame_dim + config.template_aatype_dim * 2 + 5
        self.embedding2d = Linear(input_dim, num_channels, initializer='relu')

        self.template_pair_stack = TemplatePairStack(config.template_pair_stack, global_config)
        self.output_layer_norm = nn.LayerNorm(num_channels)
        self.attention = TemplateAttention(config, global_config, query_num_channels)
        
    
    def forward(self, query_embedding:torch.Tensor, template_batch:Dict[str,torch.Tensor], mask_2d:torch.Tensor):
        template_mask = template_batch['template_mask']
        
        acts= self.feat_process(template_batch,mask_2d)
        acts=self.embedding2d(acts)
        if self.global_config.is_inference:
            torch.cuda.empty_cache()
        acts, _= self.template_pair_stack(acts, mask_2d)
        template_pair_representation = self.output_layer_norm(acts)
        embedding = checkpoint_function(
            self.attention, query_embedding, template_pair_representation, template_mask
            )
        
        return embedding

    def single_process(self, batch):
        """
          all process with float32
        """
        mask_2d=batch['mask_2d']
        dtype= mask_2d.dtype
        mask_2d=mask_2d.float()
        num_res = batch['template_aatype'].size(0)
        template_mask = batch['template_pseudo_beta_mask'].float()
        template_mask_2d = template_mask.unsqueeze(-1) * template_mask.unsqueeze(-2)

        # TODO: atoms
        template_dgram = dgram_from_positions(batch['template_pseudo_beta'].float(),
                                              **self.config.dgram_features)

        to_concat = [template_dgram, template_mask_2d[:, :, None]]
        aatype = F.one_hot(batch['template_aatype'], num_classes=22).type_as(mask_2d)
        to_concat.append(aatype.unsqueeze(0).repeat(num_res, 1, 1))
        to_concat.append(aatype.unsqueeze(1).repeat(1, num_res, 1))

        n, ca, c = [residue_constants.atom_order[a] for a in ('N', 'CA', 'C')]

        # should use float32 for affine
        rot, trans = quat_affine.make_transform_from_reference(
            n_xyz=batch['template_all_atom_positions'][:, n].float(),
            ca_xyz=batch['template_all_atom_positions'][:, ca].float(),
            c_xyz=batch['template_all_atom_positions'][:, c].float())
        
        affines = quat_affine.QuatAffine(
            quaternion=quat_affine.rot_to_quat(rot, unstack_inputs=True),
            translation=trans,
            rotation=rot,
            unstack_inputs=True)
        points = [x.unsqueeze(-2) for x in affines.translation]
        affine_vec = affines.invert_point(points, extra_dims=1)
        inv_distance_scalar = torch.rsqrt(
            1e-6 + sum([torch.square(x) for x in affine_vec]))
        
        # Backbone affine mask: whether the residue has C, CA, N
        # (the template mask defined above only considers pseudo CB).
        template_mask = (
            batch['template_all_atom_masks'][..., n] *
            batch['template_all_atom_masks'][..., ca] *
            batch['template_all_atom_masks'][..., c])
        template_mask_2d = template_mask[:, None] * template_mask[None, :]

        inv_distance_scalar = inv_distance_scalar * template_mask_2d.type_as(inv_distance_scalar)

        unit_vector = [(x * inv_distance_scalar)[..., None] for x in affine_vec]

        unit_vector = [x.type_as(mask_2d) for x in unit_vector]
        template_mask_2d = template_mask_2d.type_as(mask_2d)
        if not self.config.use_template_unit_vector:
            unit_vector = [torch.zeros_like(x) for x in unit_vector]
        to_concat.extend(unit_vector)

        to_concat.append(template_mask_2d[..., None])
        act = torch.cat(to_concat, dim=-1)

        # Mask out non-template regions so we don't get arbitrary values in the
        # distogram for these regions.
        act = act * template_mask_2d[..., None]
        # back to init dtype
        act = act.to(dtype)
        return act
    
    @torch.no_grad()
    def feat_process(self, batch:Dict[str, torch.Tensor], mask_2d:torch.Tensor):
        """
            batch: template feats with shape (bsz, NTemp, *)
            mask_2d: mask from Residue pair (bsz, NRes, NRes)
        """
        temp_batch= {k:v for k,v in batch.items() if k.startswith("template_")}
        assert len(batch['template_aatype'].shape) ==3, "batch dimmension mismatch"
        assert len(mask_2d.shape) ==3, "batch dimmension mismatch"
        bsz, ntemp= batch['template_aatype'].shape[:2]
        mask_2d= mask_2d.unsqueeze(1).repeat(1, ntemp, 1,1)
        temp_batch['mask_2d']= mask_2d
        temp_batch= {k:flatten_prev_dims(v,2) for k,v in temp_batch.items()}
        acts=[]
        
        for i in range(bsz*ntemp):
            single= {k:v[i] for k,v in temp_batch.items()}
            acts.append(self.single_process(single))
        acts= torch.stack(acts,dim=0)
        acts = acts.view((bsz,ntemp) + acts.shape[1:])
        return acts
    
