from typing import Union
import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .modules import TransformerLayer
from ..dit_module import TimestepEmbedder, LabelEmbedder, DiTBlock, FinalLayer
from ..uvit_module import trunc_normal_, Block

import logging
logger = logging.getLogger(__file__)


class LatentEmbedder(nn.Module):
    def __init__(self, in_channels, hidden_size, config) -> None:
        super().__init__()
        # assert in_channels in [config.codebook_dim, config.codebook_act_dim]
        # self.code_projection = in_channels == config.codebook_act_dim
        self.config = config
        self.in_channels = in_channels
        self.vocab_num = vocab_num = config.vocab_num

        self.wtb = nn.Embedding(vocab_num, in_channels)
        # self._init_embedding(config.pretrained_ckpt, config.norm_latent)

        self.input_activation = nn.Linear(in_channels, hidden_size)

    def forward(self, x):
        is_ids = len(x.shape) == 2
        if is_ids:
            x = self.wtb(x)
        x = self.input_activation(x)

        return x

    def _init_embedding(self, pretrained_ckpt, norm_latent):
        weights_pkl = torch.load(pretrained_ckpt, map_location='cpu')
        wtb_weights = {k.replace('codebook.codebook_layer.0.', ''):\
             v for k, v in weights_pkl['model'].items() if 'codebook.codebook_layer.0' in k}
        if norm_latent:
            self.wtb_mean = wtb_weights['weight'].mean(0)
            self.wtb_std = wtb_weights['weight'].std(0)
            normed_wtb_weights = (wtb_weights['weight'] - self.wtb_mean[None])/self.wtb_std[None]
            wtb_weights['weight'] = normed_wtb_weights

        self.wtb.load_state_dict(wtb_weights)
        

    def get_unnormed_codebook(self):
        assert self.config.norm_latent
        return (self.wtb.weight * self.wtb_std[None] )+ self.wtb_mean[None]
        


class ESM2Encoder(nn.Module):
    def __init__(
        self,
        config,
        global_config,
        in_channels, 
        out_channels=None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if (out_channels is not None) else in_channels
        self.config = config

        num_classes = config.num_classes
        class_dropout_prob = config.class_dropout_prob
        embed_dim = self.config.embed_dim
        num_layers = self.config.num_layers
        attention_heads = self.config.attention_heads

        self.x_embedder = LatentEmbedder(in_channels, embed_dim, global_config.latentembedder)
        self.t_embedder = TimestepEmbedder(embed_dim)
        self.y_embedder = LabelEmbedder(num_classes, embed_dim, class_dropout_prob)

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    embed_dim,
                    4 * embed_dim,
                    attention_heads,
                    add_bias_kv=False,
                    use_esm1b_layer_norm=True,
                    use_rotary_embeddings=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(embed_dim, eps=config.layer_norm_epsilon)
        self.out_proj = nn.Linear(embed_dim, out_channels)
        self.out_proj.weight.data.zero_()
        self.out_proj.bias.data.zero_()

        nll_head_flag = getattr(global_config.loss_weight, 'nll_loss', 0.0)
        if nll_head_flag > 0.0:
            self.nll_head = nn.Sequential(
                nn.Linear(self.out_channels, self.x_embedder.vocab_num),
                nn.GELU(),
                nn.Linear(self.x_embedder.vocab_num, self.x_embedder.vocab_num),
            )

        esm_pretrained_f = getattr(config, "esm_pretrained_f", None)
        if esm_pretrained_f is not None:
            self._esm_init_xformer(esm_pretrained_f)


    def _esm_init_xformer(self, esm_pretrained_f):
        params = torch.load(esm_pretrained_f, map_location='cpu')
        model_weights = params['model']
        esm_weight = {k.replace('encoder.sentence_encoder.layers.', ''): v for k, v in model_weights.items() \
            if (k.startswith('encoder.sentence_encoder.layers.') ) }
        self.layers.load_state_dict(esm_weight)


    def forward(
        self, x, t, y, single_mask, input_hidden=None, single_condition=None
        ):
        # (N, T, D) => (T, N, D)
        x = self.x_embedder(x if input_hidden is None else input_hidden).transpose(0, 1)                  # (T, N, D)
        if single_condition is not None:
            x = x + single_condition.transpose(0, 1)
        t = self.t_embedder(t)[None]                   # (T, N, D)
        y = self.y_embedder(y, self.training)[None]    # (T, N, D)
        x = t + y + x       # (T, N, D)

        padding_mask = 1 - single_mask
        if padding_mask is not None:
            x = x * (1 - padding_mask.transpose(0, 1).unsqueeze(-1).type_as(x))
        # import pdb; pdb.set_trace()
        for layer_idx, layer in enumerate(self.layers):
            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                x, attn = checkpoint(
                    layer, 
                    x, None, padding_mask
                )
            else:
                x, _ = layer(
                    x,
                    self_attn_padding_mask=padding_mask,
                )

        x = self.ln_f(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        x = self.out_proj(x)

        return x


class DiTEncoder(nn.Module):
    def __init__(self, config, global_config, in_channels, out_channels=None)  -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if (out_channels is not None) else in_channels
        self.config = config
        hidden_size = config.embed_dim
        num_classes = config.num_classes
        class_dropout_prob = config.class_dropout_prob
        self.num_heads = num_heads = config.attention_heads
        depth = config.depth

        self.x_embedder = LatentEmbedder(in_channels, hidden_size, global_config.latentembedder)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=4, use_rotary_embeddings=False) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, self.out_channels)

        nll_head_flag = getattr(global_config.loss_weight, 'nll_loss', 0.0)
        if nll_head_flag > 0.0:
            self.nll_head = nn.Sequential(
                nn.Linear(self.out_channels, self.x_embedder.vocab_num),
                nn.GELU(),
                nn.Linear(self.x_embedder.vocab_num, self.x_embedder.vocab_num),
            )

        self.initialize_weights()


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def forward(self, x, t, y, single_mask, input_hidden=None, single_condition=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # (N, T, D) => (T, N, D)
        x = self.x_embedder(x if input_hidden is None else input_hidden).transpose(0, 1) #+ single_condition.transpose(0, 1)                  # (T, N, D)
        if single_condition is not None:
            x = x + single_condition.transpose(0, 1)
        t = self.t_embedder(t)[None]                   # (T, N, D)
        y = self.y_embedder(y, self.training)[None]    # (T, N, D)
        c = t + y    # (T, N, D)
        # residue condition + c
        
        padding_mask = 1 - single_mask
        if padding_mask is not None:
            x = x * (1 - padding_mask.transpose(0, 1).unsqueeze(-1).type_as(x))

        for layer_idx, layer in enumerate(self.blocks):
            # if getattr(self.config, "gradient_checkpointing", False) and self.training:
            #     x, dit_learned_params = checkpoint(
            #         layer, 
            #         x, c, padding_mask
            #     )
            # else:
            #     x, dit_learned_params = layer(
            #         x, c, padding_mask,
            #     )
            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                x = checkpoint(
                    layer, 
                    x, c, padding_mask
                )
            else:
                x = layer(
                    x, c, padding_mask,
                )
            # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = dit_learned_params
            # info_list = [
            #     f'layer_{layer_idx}',
            #     'shift_msa', round(shift_msa.mean().item(), 4), round(shift_msa.std().item(), 4), round(shift_msa.min().item(), 4), round(shift_msa.max().item(), 4), 
            #     'scale_msa', round(scale_msa.mean().item(), 4), round(scale_msa.std().item(), 4), round(scale_msa.min().item(), 4), round(scale_msa.max().item(), 4), 
            #     'gate_msa',  round(gate_msa.mean().item(), 4),  round(gate_msa.std().item(), 4),  round(gate_msa.min().item(), 4),  round(gate_msa.max().item(), 4), 
            #     'shift_mlp', round(shift_mlp.mean().item(), 4), round(shift_mlp.std().item(), 4), round(shift_mlp.min().item(), 4), round(shift_mlp.max().item(), 4), 
            #     'scale_mlp', round(scale_mlp.mean().item(), 4), round(scale_mlp.std().item(), 4), round(scale_mlp.min().item(), 4), round(scale_mlp.max().item(), 4), 
            #     'gate_mlp',  round(gate_mlp.mean().item(), 4),  round(gate_mlp.std().item(), 4),  round(gate_mlp.min().item(), 4),  round(gate_mlp.max().item(), 4)
            #     ]
            # all_info = ' '.join([f'{info}' for info in info_list])
            # logger.info(all_info)
        x = self.final_layer(x, c).transpose(0, 1)     # (N, T, D)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)



class UViTEncoder(nn.Module):
    def __init__(self, config, global_config, in_channels, out_channels=None):
        super().__init__()
        self.in_channels =  self.in_chans = in_channels
        self.out_channels = out_channels if (out_channels is not None) else in_channels
        self.config = config
        self.num_features = self.embed_dim = embed_dim = config.embed_dim
        num_classes = config.num_classes
        class_dropout_prob = config.class_dropout_prob
        self.num_heads = num_heads = config.attention_heads
        depth = config.depth
        qkv_bias = getattr(config, "qkv_bias", False)
        qk_scale = getattr(config, "qk_scale", False)
        norm_layer = getattr(config, "norm_layer", nn.LayerNorm)
        skip = getattr(config, "skip", True)
        mlp_ratio = getattr(config, "mlp_ratio", 4)
        use_checkpoint = getattr(config, "use_checkpoint", True)

        self.x_embedder = LatentEmbedder(in_channels, embed_dim, global_config.latentembedder)
        self.t_embedder = TimestepEmbedder(embed_dim)
        self.y_embedder = LabelEmbedder(num_classes, embed_dim, class_dropout_prob)
        self.extras = 2
        self.pos_embed = nn.Parameter(torch.zeros(self.extras, 1, self.embed_dim))

        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.mid_block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)

        self.out_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.ln_f = nn.LayerNorm(embed_dim, eps=config.layer_norm_epsilon)
        self.out_proj = nn.Linear(embed_dim, out_channels)
        self.out_proj.weight.data.zero_()
        self.out_proj.bias.data.zero_()

        self.apply(self._init_weights)
        trunc_normal_(self.pos_embed, std=.02)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}


    def forward(
        self, x, t, y, single_mask, input_hidden=None, single_condition=None
        ):
        batchsize, res_num = single_mask.shape[:2]
        # (N, T, D) => (T, N, D)
        x = self.x_embedder(x if input_hidden is None else input_hidden).transpose(0, 1)  # (T, N, D)
        if single_condition is not None:
            x = x + single_condition.transpose(0, 1)
        padding_mask = 1 - single_mask
        if padding_mask is not None:
            x = x * (1 - padding_mask.transpose(0, 1).unsqueeze(-1).type_as(x))

        t = self.t_embedder(t)[None]                   # (T, N, D)
        y = self.y_embedder(y, self.training)[None]    # (T, N, D)
        x = x + y + t
        # extra_s = torch.cat((y, t), dim=0)
        # extra_s = extra_s + self.pos_embed

        # x = torch.cat((extra_s, x), dim=0) # (T+2, N, D)
        # padding_mask = torch.cat([torch.zeros(batchsize, 2).type_as(padding_mask), padding_mask], 1)
        

        skips = []
        for blk in self.in_blocks:
            x = blk(x, self_attn_padding_mask=padding_mask)
            skips.append(x)

        x = self.mid_block(x, self_attn_padding_mask=padding_mask)

        for blk in self.out_blocks:
            x = blk(x, skips.pop(), self_attn_padding_mask=padding_mask)

        # x = self.ln_f(x)[self.extras:] # (T+2, B, E) -> (T, B, E)
        x = self.ln_f(x)# (T+2, B, E) -> (T, B, E)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        x = self.out_proj(x)
        return x