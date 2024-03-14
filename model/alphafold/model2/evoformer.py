import torch
from torch import nn
from torch.nn import functional as F
# from torch.utils.checkpoint import checkpoint,checkpoint_sequential
from .utils import checkpoint, checkpoint_sequential, MultiArgsSequential, ResModule
from typing import Optional,List, Dict
from .layers import *
from .template import TemplateEmbedding
from alphafold import all_atom


    
def build_block(
    config,
    global_config,
    msa_channel,
    pair_channel,
    is_extra_msa
):
    msa_dropout= config.msa_row_attention_with_pair_bias.dropout_rate
    pair_dropout_row= config.triangle_attention_starting_node.dropout_rate
    if global_config.deterministic:
        msa_dropout= pair_dropout_row= 0.
    modules=[]
    
    if config.outer_product_mean.first:
        modules.append(
            ResModule(
                OuterProductMean(
                    config.outer_product_mean, global_config, msa_channel, pair_channel
                ),
                None,input_indices=(0,2),output_index=1,
                name= "outer_product_mean"
            )
        )
    modules.append(
        ResModule(
            MSARowAttentionWithPairBias(
                config.msa_row_attention_with_pair_bias, global_config, msa_channel, pair_channel
            ),
            DropoutRowwise(msa_dropout),
            input_indices=(0,2,1),output_index=0,
            name= "msa_row_attention_with_pair_bias"
        )
    )
    if not is_extra_msa:
        modules.append(
            ResModule(
                MSAColumnAttention(
                    config.msa_column_attention, global_config, msa_channel,
                ),
                None,input_indices=(0,2),output_index=0,
                name= "msa_column_attention"
            )
        )
    else:
        modules.append(
            ResModule(
                MSAColumnGlobalAttention(
                config.msa_column_attention, global_config, msa_channel
                ),
                None,input_indices=(0,2),output_index=0,
                name= "msa_column_global_attention"
            )
        )
    modules.append(
        ResModule(
            Transition(
                config.msa_transition, global_config, msa_channel
            ),
            None,
            input_indices=(0,2),output_index=0,
            name = "msa_transition"
        )
    )
    if not config.outer_product_mean.first:
        modules.append(
            ResModule(
                OuterProductMean(
                    config.outer_product_mean, global_config, msa_channel, pair_channel
                ),
                None,input_indices=(0,2),output_index=1,
                name= "outer_product_mean"
            )
        )
    modules.append(
        ResModule(
            TriangleMultiplication(
                config.triangle_multiplication_outgoing, global_config, pair_channel
            ),
            DropoutRowwise(pair_dropout_row),
            input_indices=(1,3),output_index=1,
            name = "triangle_multiplication_outgoing"
        )
    )
    modules.append(
        ResModule(
            TriangleMultiplication(
                config.triangle_multiplication_incoming, global_config, pair_channel
            ),
            DropoutRowwise(pair_dropout_row),
            input_indices=(1,3),output_index=1,
            name= "triangle_multiplication_incoming"
        )
    )
    modules.append(
        ResModule(
            TriangleAttention(
                config.triangle_attention_starting_node, global_config, pair_channel
            ),
            DropoutRowwise(pair_dropout_row),
            input_indices=(1,3),output_index=1,
            name= "triangle_attention_starting_node"
        )
    )
    modules.append(
        ResModule(
            TriangleAttention(
                config.triangle_attention_ending_node, global_config, pair_channel
            ),
            DropoutColumnwise(pair_dropout_row),
            input_indices=(1,3),output_index=1,
            name= "triangle_attention_ending_node"
        )
    )
    modules.append(
        ResModule(
            Transition(
                config.pair_transition, global_config, pair_channel
            ),
            None,
            input_indices=(1,3),output_index=1,
            name= "pair_transition"
        )
    )
    return modules
    
class EvoEmbedding(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.config = config
        self.global_config = global_config

        self.preprocess_1d = Linear(config.target_feat_dim, config.msa_channel)   # 22
        self.preprocess_msa = Linear(config.msa_feat_dim, config.msa_channel)  # 49

        self.left_single = Linear(config.target_feat_dim, config.pair_channel)
        self.right_single = Linear(config.target_feat_dim, config.pair_channel)

        self.prev_pos_linear = Linear(config.prev_pos.num_bins, config.pair_channel)
        self.pair_activiations = Linear(2 * config.max_relative_feature + 1, config.pair_channel)

        self.prev_msa_first_row_norm = nn.LayerNorm(config.msa_channel)
        self.prev_pair_norm = nn.LayerNorm(config.pair_channel)

    def forward(self, batch):
        c = self.config
        gc = self.global_config
        
        preprocess_1d = self.preprocess_1d(batch['target_feat'])
        preprocess_msa = self.preprocess_msa(batch['msa_feat'])

        msa_activations = preprocess_1d.unsqueeze(-3) + preprocess_msa
        
        left_single = self.left_single(batch['target_feat'])
        right_single = self.right_single(batch['target_feat'])
        pair_activations = left_single[...,None,:] + right_single[...,None,:,:]

        # Inject previous outputs for recycling.
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" line 6
        # Jumper et al. (2021) Suppl. Alg. 32 "RecyclingEmbedder"
        
        if c.recycle_pos and 'prev_pos' in batch:
            prev_pseudo_beta = pseudo_beta_fn(
                batch['aatype'], batch['prev_pos'], None)
            dgram = dgram_from_positions(prev_pseudo_beta, **self.config.prev_pos)
            prev_pseudo_beta=prev_pseudo_beta.to(pair_activations.dtype)
            dgram= dgram.to(pair_activations.dtype)
            pair_activations = pair_activations + self.prev_pos_linear(dgram)

        # recycle_features
        
        if c.recycle_features:
            if 'prev_msa_first_row' in batch:
                msa_activations[:,0] = msa_activations[:,0] + self.prev_msa_first_row_norm(batch['prev_msa_first_row'])
            if 'prev_pair' in batch:
                pair_activations = pair_activations + self.prev_pair_norm(batch['prev_pair'])

        # pos embedding
        if c.max_relative_feature:
            pos = batch['residue_index']
            
            offset = pos[...,:, None] - pos[...,None, :]
            
            rel_pos = torch.clamp(
                offset + c.max_relative_feature,
                min = 0,
                max = 2 * c.max_relative_feature
            )
            rel_pos = F.one_hot(rel_pos.long(), num_classes=2 * c.max_relative_feature + 1)
            pair_activations = pair_activations + self.pair_activiations(rel_pos.to(pair_activations))
        return pair_activations, msa_activations


class EmbeddingsAndEvoformer(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.evo_emb= EvoEmbedding(config, global_config)
        self.template_embedding = TemplateEmbedding(config.template, global_config, config.pair_channel)
        self.extra_msa_activations = Linear(config.extra_msa_feat_dim, config.extra_msa_channel)
        extra_msa_list =[]
        for _ in range(config.extra_msa_stack_num_block):
            extra_msa_list.extend(
                build_block(
                    config.evoformer, global_config,
                    config.extra_msa_channel, config.pair_channel, is_extra_msa=True
                )
            )
        self.extra_msa_stack = MultiArgsSequential(* extra_msa_list)
        evoiter_list=[]
        for _ in range(config.evoformer_num_block):
            evoiter_list.extend(
                build_block(
                    config.evoformer, global_config,
                    config.msa_channel, config.pair_channel, is_extra_msa=False
                )
            )
        self.evoformer_iteration = MultiArgsSequential(*evoiter_list)
        self.template_ffn= nn.Sequential(
            Linear(config.template_feat_dim, config.msa_channel, initializer='relu'), # 57
            nn.ReLU(),
            Linear(config.msa_channel, config.msa_channel, initializer='relu')
        )
        self.single_activations = Linear(config.msa_channel, config.seq_channel)
    
    def forward(self, batch):
        c = self.config
        gc = self.global_config
        pair_activations,msa_activations = self.evo_emb(batch)
        if self.global_config.is_inference:
            torch.cuda.empty_cache()
        mask_2d = batch['seq_mask'][...,:, None] * batch['seq_mask'][...,None, :]
        
        if c.template.enabled:
            template_batch = {k: batch[k] for k in batch if k.startswith('template_')}
            template_pair_representation = self.template_embedding(
                pair_activations, template_batch, mask_2d
            )
            pair_activations = pair_activations + template_pair_representation
        
        # extra MSA features
        extra_msa_feat = create_extra_msa_feature(batch)
        extra_msa_activations = self.extra_msa_activations(extra_msa_feat)
        # Extra MSA Stack.
        # Jumper et al. (2021) Suppl. Alg. 18 "ExtraMsaStack"
        extra_msa_input = extra_msa_activations
        extra_pair_input = pair_activations
        # for layer in self.extra_msa_stack:
        #     extra_msa_input, extra_pair_input, _, _= layer(
        #         extra_msa_input, extra_pair_input, batch['extra_msa_mask'], mask_2d
        #         )
        # print(f"extra_pair_input shape {extra_pair_input.shape}")
        # for layer in self.extra_msa_stack:
        #     extra_msa_input, extra_pair_input, _, _  = layer(
        #         extra_msa_input, extra_pair_input, batch['extra_msa_mask'], mask_2d
        #     )
        #     print(f"layer {layer},extra_pair_input shape {extra_pair_input.shape}")
        # for n,layer in enumerate(self.extra_msa_stack):
        #     extra_msa_input, extra_pair_input, _, _=layer(extra_msa_input, extra_pair_input, batch['extra_msa_mask'], mask_2d)
        #     if torch.isnan(extra_msa_input.std()) or torch.isnan(extra_pair_input.std()):
        #         print(f'layer {n}, {layer}')
        #         import pdb;pdb.set_trace()
                
        
        extra_msa_input, extra_pair_input, _, _ =checkpoint_sequential(
            self.extra_msa_stack,
            c.extra_msa_checkpoint,
            (extra_msa_input, extra_pair_input, batch['extra_msa_mask'], mask_2d)
        )
        pair_activations = extra_pair_input

        evoformer_input = {
            'msa': msa_activations,
            'pair': pair_activations,
        }

        evoformer_masks = {'msa': batch['msa_mask'], 'pair': mask_2d}

        # Append num_templ rows to msa_activations with template embeddings.
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 7-8
        if c.template.enabled and c.template.embed_torsion_angles:
            batch_size, num_templ, num_res = batch['template_aatype'].shape
            dtype=batch['template_all_atom_masks'].dtype
            # Embed the templates aatypes.
            with torch.no_grad():
                aatype_one_hot = F.one_hot(batch['template_aatype'], num_classes=22)
                ret = all_atom.atom37_to_torsion_angles(
                    aatype=flatten_prev_dims( batch['template_aatype'],2),
                    all_atom_pos=flatten_prev_dims(batch['template_all_atom_positions'],2),
                    all_atom_mask=flatten_prev_dims(batch['template_all_atom_masks'],2),
                    # Ensure consistent behaviour during testing:
                    placeholder_for_undefined=not gc.zero_init)

            template_features = torch.cat([
                aatype_one_hot,
                ret['torsion_angles_sin_cos'].view(batch_size, num_templ, num_res, 14),
                ret['alt_torsion_angles_sin_cos'].view(batch_size, num_templ, num_res, 14),
                ret['torsion_angles_mask'].view(batch_size,num_templ, num_res, -1)], dim=-1)
            template_features= template_features.to(dtype)
            
            template_activations = self.template_ffn(template_features)

            # Concatenate the templates to the msa.
            evoformer_input['msa'] = torch.cat(
                [evoformer_input['msa'], template_activations], dim=-3
            )
            # Concatenate templates masks to the msa masks.
            # Use mask from the psi angle, as it only depends on the backbone atoms
            # from a single residue.
            
            torsion_angle_mask = ret['torsion_angles_mask']\
                .view(batch_size,num_templ, num_res, -1)[..., 2]
            torsion_angle_mask = torsion_angle_mask.type_as(evoformer_masks['msa'])
            evoformer_masks['msa'] = torch.cat(
                [evoformer_masks['msa'], torsion_angle_mask], dim=-2)

        
        evoiter_out = checkpoint_sequential(
            self.evoformer_iteration,
            c.evo_former_checkpoint,
            (evoformer_input['msa'], evoformer_input['pair'],
                evoformer_masks['msa'],evoformer_masks['pair'])
        )
        
    
        evoformer_output = {
            'msa': evoiter_out[0],
            'pair': evoiter_out[1]
        }

        msa_activations = evoformer_output['msa']
        pair_activations = evoformer_output['pair']

        single_activations = self.single_activations(msa_activations[:,0])
        num_sequences = batch['msa_feat'].shape[-3]
        output = {
            'single': single_activations,
            'pair': pair_activations,
            # Crop away template rows such that they are not used in MaskedMsaHead.
            'msa': msa_activations[...,:num_sequences, :, :],
            'msa_first_row': msa_activations[:,0],
        }

        return output
