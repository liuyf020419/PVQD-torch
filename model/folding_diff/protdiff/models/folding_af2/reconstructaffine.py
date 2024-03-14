import os
os.sys.path.append("/yrfs1/intern/yfliu25/protein_diffusion/models")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .evo_diff import *
from .folding_batch import *
from dense_block import *


class ReconstructAffine(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()

        self.config = config
        self.diffevoformer = DiffEvoformer(config.diffevoformer, global_config)
        self.affine_gen = AffineGenerator_nSC(config.structure_module,
                                                global_config,
                                                config.structure_module.seq_channel,
                                                 config.structure_module.pair_channel
                                                )
        self.continous_noise = ContinousNoiseSchedual(config.structure_module.noise_channel)

        if config.structure_module.pair_updated:
            self.single_out = Linear(config.structure_module.seq_channel, config.diffevoformer.single_target_dim)
            self.pair_out = Linear(config.structure_module.pair_channel, config.diffevoformer.pair_target_dim)


    def forward(self, geom_maps, BB_tors, quataffine, cnoise):
        representation = self.prepare_repr(geom_maps, BB_tors, quataffine)
        
        assert all(np.isin(["single", "residue_index", "pair", "affine"], list(representation.keys())))

        cnoise_emb = self.continous_noise(cnoise)
        act_representation = self.diffevoformer(representation, cnoise_emb)
        act_representation["affine"] = representation["affine"]
        # import pdb; pdb.set_trace()
        act_affine = self.affine_gen(act_representation, cnoise_emb)

        if not self.config.structure_module.pair_updated:
            return act_affine["affine"][-1]

        else:
            single_out = self.single_out(act_affine["act"])
            pair_out = self.pair_out(act_affine["pair"]).permute([0, 3, 1, 2])
            return act_affine["affine"][-1], single_out, pair_out


    def prepare_repr(self, geom_maps, BB_tors, quataffine):
        residue_index = torch.arange(BB_tors.size(1))
        representation = {"single": BB_tors, "pair": geom_maps, 
                          "residue_index": residue_index, "affine": quataffine}

        return representation
