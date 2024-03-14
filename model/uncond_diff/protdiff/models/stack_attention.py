from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from unifold.modules.evoformer import EvoformerStack
from unifold.modules.attentions import (
    gen_msa_attn_mask,
    gen_tri_attn_mask,
    MSARowAttentionWithPairBias,
    MSAAttention
)
from .encoder_module.attention.modules import TransformerLayer
from .framediff.framediff_module import IPAAttention
from .nn_utils import generate_new_affine

import flash_attn


class SingleToPairModule(nn.Module):
    def __init__(self, config, global_config, single_in_dim, pair_out_dim) -> None:
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.single_channel = single_channel = self.config.single_channel
        self.pair_out_dim = pair_out_dim

        self.layernorm = nn.LayerNorm(single_in_dim)
        self.single_act = nn.Linear(single_in_dim, single_channel)
        self.pair_act = nn.Linear(single_channel, self.pair_out_dim)

    def forward(self, single):
        single = self.layernorm(single)
        single_act = self.single_act(single)

        q, k = single_act.chunk(2, -1)
        prod = q[:, None, :, :] * k[:, :, None, :]
        diff = q[:, None, :, :] - k[:, :, None, :]
        pair = torch.cat([prod, diff], -1)
        pair_act = self.pair_act(pair)

        return pair_act



class TransformerStackDecoder(nn.Module):
    def __init__(self, config, global_config, single_in_dim, with_bias=False, out_dim=None, layer_num=None) -> None:
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.with_bias = with_bias
        self.encode_pair_idx = self.global_config.encode_pair_idx
        self.single_channel = single_channel = self.config.single_channel
        if out_dim is None:
            out_dim = single_channel
        if layer_num is not None:
            layer_num = layer_num
        else:
            layer_num = config.layers

        self.layernorm = nn.LayerNorm(single_in_dim)
        self.single_act = nn.Linear(single_in_dim, single_channel)
        if with_bias:
            self.pair_act = nn.Linear(single_channel, single_channel)
            if self.encode_pair_idx:
                self.pair_position_embedding = nn.Embedding(
                    self.global_config.pair_res_range[1] * 2 + 2, single_channel)
        if with_bias:
            self.attention = nn.ModuleList(
                [
                    MSARowAttentionWithPairBias(
                        d_msa=config.single_channel,
                        d_pair=config.single_channel,
                        d_hid = config.ffn_embed_dim//config.attention_heads,
                        num_heads = config.attention_heads,
                    )
                    for _ in range(layer_num)
                ]
            )

        else:
            self.attention = nn.ModuleList(
                [
                    MSAAttention(
                        config.single_channel,
                        config.ffn_embed_dim//config.attention_heads,
                        config.attention_heads,
                    )
                    for _ in range(layer_num)
                ]
            )

        self.out_layer = nn.Linear(config.single_channel, out_dim)


    def forward(self, single, single_mask, pair_idx=None, pair_init=None):
        msa_mask = single_mask[:, None]
        msa_row_mask, msa_col_mask = gen_msa_attn_mask(
            msa_mask,
            inf=self.inf,
        )

        m = self.layernorm(single)
        single_act = self.single_act(m)[:, None]
        if self.with_bias:
            q, k = single_act.chunk(2, -1)
            prod = q[:, None, :, :] * k[:, :, None, :]
            diff = q[:, None, :, :] - k[:, :, None, :]
            pair = torch.cat([prod, diff], -1)
            pair_act = self.pair_act(pair)
            if self.encode_pair_idx:
                assert pair_idx is not None
                pair_act = pair_act + self.pair_position_embedding(pair_idx)
            if pair_init is not None:
                pair_act = pair_act + pair_init

        single_post = self.attention(
            single_act,
            pair_act if self.with_bias else None,
            msa_row_mask
        )
        single_post = self.out_layer(single_post)

        return single_post



class EvoformerStackDecoder(nn.Module):
    def __init__(self, config, global_config, single_in_dim, out_dim=None) -> None:
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.encode_pair_idx = self.global_config.encode_pair_idx
        self.single_channel = single_channel = self.config.evoformer_stack.d_msa
        self.pair_channel = pair_channel = self.config.evoformer_stack.d_pair
        if out_dim is None:
            out_dim = single_channel
        
        self.layernorm = nn.LayerNorm(single_in_dim)
        self.single_act = nn.Linear(single_in_dim, single_channel)

        self.pair_act = nn.Linear(single_channel, pair_channel)
        if self.encode_pair_idx:
            self.pair_position_embedding = nn.Embedding(
                self.global_config.pair_res_range[1] * 2 + 2, pair_channel)
        self.inf = 3e4
        self.evoformer = EvoformerStack(**self.config.evoformer_stack)
        self.out_layer = nn.Linear(config.evoformer_stack.d_single, out_dim)


    def forward(self, single, single_mask, pair_idx=None, pair_init=None):
        pair_mask = single_mask[:, None] * single_mask[:, :, None]
        tri_start_attn_mask, tri_end_attn_mask = gen_tri_attn_mask(pair_mask, self.inf)

        msa_mask = single_mask[:, None]
        msa_row_mask, msa_col_mask = gen_msa_attn_mask(
            msa_mask,
            inf=self.inf,
        )
        single = self.layernorm(single)
        single = self.single_act(single)
        m = single[:, None]

        q, k = single.chunk(2, -1)
        prod = q[:, None, :, :] * k[:, :, None, :]
        diff = q[:, None, :, :] - k[:, :, None, :]
        pair = torch.cat([prod, diff], -1)
        # import pdb; pdb.set_trace()
        z = self.pair_act(pair)
        if self.encode_pair_idx:
            assert pair_idx is not None
            z = z + self.pair_position_embedding(pair_idx)

        if pair_init is not None:
            z = z + pair_init

        m, z, s = self.evoformer(
            m,
            z,
            msa_mask=msa_mask,
            pair_mask=pair_mask,
            msa_row_attn_mask=msa_row_mask,
            msa_col_attn_mask=msa_col_mask,
            tri_start_attn_mask=tri_start_attn_mask,
            tri_end_attn_mask=tri_end_attn_mask,
            chunk_size=self.config.globals.chunk_size,
            block_size=self.config.globals.block_size,
        )
        # import pdb; pdb.set_trace()
        assert(len(s.shape) == 4 or len(s.shape) == 3)
        if len(s.shape) == 4:
            s = s[:, 0]
        single_rep = s
        pair_rep = z
        # import pdb; pdb.set_trace()
        single_rep = self.out_layer(single_rep)

        return single_rep, pair_rep



class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight



def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis



def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)



def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    # import pdb; pdb.set_trace()
    if (len(freqs_cis.shape) ==2):
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    else:
        assert len(freqs_cis.shape) == 3
        freqs_cis = freqs_cis[:, :, None]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)



class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.n_local_heads = args.n_heads # // fs_init.get_model_parallel_world_size()
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
        )


    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        xq = xq.transpose(1, 2)
        keys = xk.transpose(1, 2)
        values = xv.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)



class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        # hidden_dim = int(2 * hidden_dim / 3)
        # hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))



class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        # import pdb; pdb.set_trace()
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out



class TransformerRotary(nn.Module):
    def __init__(self, config, global_config, single_in_dim, out_dim=None, layer_num=None):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.n_layers = config.n_layers
        self.single_channel = single_channel = config.dim
        if out_dim is None:
            out_dim = single_channel
        if layer_num is not None:
            layer_num = layer_num
        else:
            layer_num = config.n_layers

        self.layernorm = nn.LayerNorm(single_in_dim)
        self.single_act = nn.Linear(single_in_dim, single_channel)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(layer_num):
            self.layers.append(TransformerBlock(layer_id, config))

        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        self.freqs_cis = precompute_freqs_cis(
            self.config.dim // self.config.n_heads, self.config.max_seq_len * 2
        )

        self.out_layer = nn.Linear(single_channel, out_dim)

        if (self.global_config.aatype_embedding and self.aatype_embedding_in_outlayer):
            self.aatype_embedding = nn.Sequential(
                nn.Embedding(22, single_channel),
                nn.ReLU(),
                nn.LayerNorm(single_channel),
                nn.Linear(single_channel, out_dim))


    def forward(self, x: torch.Tensor, single_idx: torch.Tensor, single_mask: torch.Tensor=None, pair_mask: torch.Tensor=None, start_pos: int=0, input_aatype=None, batch=None):
        _bsz, seqlen = x.shape[:2]

        self.freqs_cis = self.freqs_cis.to(x.device)
        cis_dim = self.freqs_cis.shape[-1]
        # freqs_cis = self.freqs_cis[:seqlen]
        paded_single_idx = single_idx + (1 - single_mask) * (self.config.max_seq_len-1)
        freqs_cis = self.freqs_cis[paded_single_idx.reshape(-1, 1).long()].reshape(_bsz, seqlen, cis_dim)

        mask = None
        if single_mask is not None:
            s_mask = (single_mask - 1.) * -2e15
            mask = -(s_mask[:, None] * s_mask[:, :, None])[:, None]
        if pair_mask is not None:
            mask = (1. - pair_mask)[:, None] * -2e15 + mask

        x = self.layernorm(x)
        x = self.single_act(x)
        for layer in self.layers:
            x = layer(x, start_pos, freqs_cis, mask)
        x = self.norm(x)

        x = self.out_layer(x)

        if (self.global_config.aatype_embedding and self.aatype_embedding_in_outlayer):
            # import pdb; pdb.set_trace()
            assert input_aatype is not None
            if not self.training:
                aatype_drop_p = 0.0
            else:
                aatype_drop_p = self.global_config.aatype_drop_p
            aatype_mask = (torch.rand_like(single_mask) > aatype_drop_p).float() * single_mask
            batch['aatype_mask'] = (1 - aatype_mask) * single_mask
            aatype = (input_aatype * aatype_mask + (1 - aatype_mask) * 21).long()
            x = self.aatype_embedding(aatype) + x

        return x



class IPAattentionStackedDecoder(nn.Module):
    def __init__(self, config, global_config, single_in_dim, out_dim=None) -> None:
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.encode_pair_idx = self.global_config.encode_pair_idx
        self.single_channel = single_channel = self.config.ipa.c_s
        self.pair_channel = pair_channel = self.config.ipa.c_z
        if out_dim is None:
            out_dim = single_channel

        self.layernorm = nn.LayerNorm(single_in_dim)
        self.single_act = nn.Linear(single_in_dim, single_channel)

        preprocess_config = self.config.preprocess_layer
        self.preprocess_layers = nn.ModuleList(
            [
                TransformerLayer(
                    single_channel,
                    preprocess_config.ffn_embed_dim,
                    preprocess_config.attention_heads,
                    dropout = getattr(preprocess_config, 'dropout', 0.0),
                    add_bias_kv=True,
                    use_esm1b_layer_norm=False,
                )
                for _ in range(preprocess_config.layers)
            ]
        )

        self.single_pre_layernorm = nn.LayerNorm(single_channel)
        self.single_pre_act = nn.Linear(single_channel, single_channel)

        self.pair_act = nn.Linear(single_channel, pair_channel)
        if self.encode_pair_idx:
            self.pad_pair_res_num = self.global_config.pair_res_range[1] * 2 + 1
            self.pad_pair_chain_num = self.global_config.pair_chain_range[1] * 2 + 1
            self.pair_res_embedding = nn.Embedding(
                self.pad_pair_res_num + 1, pair_channel)
            self.pair_chain_embedding = nn.Embedding(
                self.pad_pair_chain_num + 1, pair_channel)
            self.pair_chain_entity_embedding = nn.Embedding(2 + 1, pair_channel)

        self.ipa_attention = IPAAttention(self.config)

        self.out_layer = nn.Linear(single_channel, out_dim)


    def forward(self, single, single_mask, pair_res_idx=None, pair_chain_idx=None, pair_same_entity=None, pair_init=None):
        single = self.layernorm(single)
        single = self.single_act(single)

        padding_mask = 1.0 -single_mask
        if not padding_mask.any():
            padding_mask = None

        for layer in self.preprocess_layers:
            single = single.transpose(0, 1)
            single, attn = layer(single, self_attn_padding_mask=padding_mask)
            single = single.transpose(0, 1)
        single = single * single_mask[..., None]

        single = self.single_pre_layernorm(single)
        single = self.single_pre_act(single)

        q, k = single.chunk(2, -1)
        prod = q[:, None, :, :] * k[:, :, None, :]
        diff = q[:, None, :, :] - k[:, :, None, :]
        pair = torch.cat([prod, diff], -1)

        pair = self.pair_act(pair)
        if self.encode_pair_idx:
            assert pair_res_idx is not None
            pair_pad = (1 - (single_mask[:, None] * single_mask[:, :, None])) 
            pair_res_idx = (pair_res_idx + pair_pad * self.pad_pair_res_num ).long()
            pair_chain_idx = (pair_chain_idx + pair_pad * self.pad_pair_chain_num ).long()
            pair_same_entity = (pair_same_entity + pair_pad * 2).long()

            pair = pair + self.pair_res_embedding(pair_res_idx) + \
                self.pair_chain_embedding(pair_chain_idx) + self.pair_chain_entity_embedding(pair_same_entity)

        if pair_init is not None:
            pair = pair + pair_init

        zero_affine = generate_new_affine(single_mask, return_frame=False)
        model_out = self.ipa_attention(single, pair, single_mask, zero_affine)

        single_out = self.out_layer(model_out['curr_node_embed'])

        return model_out['curr_affine'], single_out, model_out['curr_edge_embed']