import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..encoder_module.gvpstructure_embedding import GVPStructureEmbedding
from ..encoder_module.attention.modules import TransformerLayer


class GVPConditioner(nn.Module):
    def __init__(self, config, global_config) -> None:
        super().__init__()
        self.config = config.gvp_conditioner
        self.global_config = global_config

        self.encoder = GVPStructureEmbedding(
                self.config.gvp_embedding, self.global_config)
        self.embed_dim = embed_dim = self.encoder.embed_dim
        self.sstype_embedding = nn.Embedding(4+1, embed_dim)
        self.contact_onehot_embedding = nn.Embedding(2+1, embed_dim)
        
        self.layernorm = nn.LayerNorm(embed_dim)
        self.single_act = nn.Linear(embed_dim, embed_dim)

        self.preprocess_config = preprocess_config = self.config.preprocess_layer
        self.preprocess_layers = nn.ModuleList(
            [
                TransformerLayer(
                    embed_dim,
                    embed_dim * 4,
                    preprocess_config.attention_heads,
                    dropout = getattr(preprocess_config, 'dropout', 0.0),
                    add_bias_kv=True,
                    use_esm1b_layer_norm=False,
                )
                for _ in range(preprocess_config.layers)
            ]
        )

        if getattr(global_config.loss_weight, "sidechain_embed_loss", 0.0) > 0.0:
            self.sc_condtion_rep_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, global_config.in_channels),
            )
        if getattr(global_config.loss_weight, "sidechain_simi_loss", 0.0) > 0.0:
            self.sc_condtion_rep_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, global_config.in_channels),
            )

    def forward(self, batch, mixed_nn=False):
        single_mask = batch['single_mask']
        condition_mask = batch['condition_mask']
        # import pdb; pdb.set_trace()
        encoded_feature = self.encoder(batch, condition_mask, mixed_nn=mixed_nn)
        ligand_sstype = (batch['sstype'] * batch['receptor_mask']).long() # ligand only
        receptor_contact_onehot = (batch['contact_onehot'] == 2).long() # receptor only
        sstype_embed = self.sstype_embedding(ligand_sstype)
        contact_onehot_embed = self.contact_onehot_embedding(receptor_contact_onehot)
        encoded_feature = (encoded_feature * condition_mask[..., None] + sstype_embed + contact_onehot_embed) * single_mask[..., None]
        single = self.layernorm(encoded_feature)
        single = self.single_act(single)

        padding_mask = 1.0 -single_mask
        if not padding_mask.any():
            padding_mask = None

        for layer in self.preprocess_layers:
            x = single.transpose(0, 1)
            if getattr(self.preprocess_config, "gradient_checkpointing", False) and self.training:
                x, _ = checkpoint(
                    layer, 
                    x, None, padding_mask
                )
            else:
                x, _ = layer(
                    x, self_attn_padding_mask=padding_mask,
                )
            single = x.transpose(0, 1)
        single = single * single_mask[..., None]

        if (getattr(self.global_config.loss_weight, "sidechain_embed_loss", 0.0) > 0.0 or getattr(self.global_config.loss_weight, "sidechain_simi_loss", 0.0) > 0.0):
            sc_condtion_rep = self.sc_condtion_rep_head(single)
            return single, sc_condtion_rep

        return single