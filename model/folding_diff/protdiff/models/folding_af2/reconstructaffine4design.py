import os
os.sys.path.append("/yrfs1/intern/yfliu25/protein_diffusion/models")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .evo_diff import *
from .folding_batch import *
from dense_block import *

from .common.residue_constants import rigid_group_bb_positions_mean
from .quat_affine import *
from .structure_2d import get_backbone_tors_batch, extract_2d_maps_batch


rigid_group_bb_positions_mean = torch.Tensor(list(rigid_group_bb_positions_mean.values()))



def save_pdb_file(ca_crd, chain="A", filename='testloop.pdb'):
    from Bio.PDB.StructureBuilder import StructureBuilder
    from Bio.PDB import PDBIO
    from Bio.PDB.Atom import Atom

    sb = StructureBuilder()
    sb.init_structure("pdb")
    sb.init_seg(" ")
    sb.init_model(0)
    chain_id = chain
    sb.init_chain(chain_id)
    for num, line in enumerate(ca_crd):

        name = 'CA'

        line = np.around(np.array(line, dtype='float'), decimals=3)

        atom = Atom(name=name, coord=line, element=name[0:1], bfactor=1, occupancy=1, 
                    fullname=name, serial_number=num, altloc=' ')

        sb.init_residue("DUM", " ", num, " ")  # Dummy residue
        sb.structure[0][chain_id].child_list[num].add(atom.copy())

    structure = sb.structure
    io = PDBIO()
    io.set_structure(structure)
    io.save(filename)


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


    def forward(self, quataffine, cnoise):
        representation = self.prepare_repr(quataffine)
        
        # print([torch.any(torch.isnan(representation[k])) for k in representation.keys()])

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


    def prepare_repr(self, quataffine, encode_tors=True):

        globalcrds = self.affine2globalcrd(quataffine, scaler=self.config.structure_module.scale_factor)
        B, N_r, N_a, _ = globalcrds.shape

        res_mask = torch.ones(B, N_r, N_a).to(quataffine.device)
        ## residue type are assigned as gly acccordidng to AF2
        aatype = torch.Tensor(N_r * [7]).to(quataffine.device)

        BB_tors = get_backbone_tors_batch(globalcrds, aatype)
        geom_maps = extract_2d_maps_batch(globalcrds, res_mask, aatype)
        # print([torch.any(torch.isnan(gmp)) for gmp in torch.stack(list(geom_maps.values()), 0)])
        # import pdb; pdb.set_trace()
        # geom_maps["Dist"] = torch.where(geom_maps["Dist"] >= 20/self.config.structure_module.scale_factor, torch.ones_like(geom_maps["Dist"])*20, geom_maps["Dist"])
        geom_maps = torch.stack(list(geom_maps.values()), -1)
        B, N_r, _, _ = geom_maps.shape
        BB_tors = torch.stack(list(BB_tors.values()), -1)

        if encode_tors:
            encoded_geom_maps = torch.ones([B, N_r, N_r, 6])
            encoded_geom_maps[..., 0] = (geom_maps[..., 0] / 10)
            encoded_geom_maps[..., 1] = torch.sin(geom_maps[..., 1])
            encoded_geom_maps[..., 2] = torch.cos(geom_maps[..., 1])
            encoded_geom_maps[..., 3] = torch.sin(geom_maps[..., 2])
            encoded_geom_maps[..., 4] = torch.cos(geom_maps[..., 2])
            encoded_geom_maps[..., 5] = (2 * geom_maps[..., 3] / math.pi) - 1

            # mask_gms = (encoded_geom_maps[..., 0] == 1)
            
            # encoded_geom_maps[..., 1] = torch.where(mask_gms, 1, encoded_geom_maps[..., 1])
            # encoded_geom_maps[..., 2] = torch.where(mask_gms, 0, encoded_geom_maps[..., 2])
            # encoded_geom_maps[..., 3] = torch.where(mask_gms, 1, encoded_geom_maps[..., 3])
            # encoded_geom_maps[..., 4] = torch.where(mask_gms, 0, encoded_geom_maps[..., 4])
            # encoded_geom_maps[..., 5] = torch.where(mask_gms, 1, encoded_geom_maps[..., 5])

            encoded_geom_maps = encoded_geom_maps.permute(0, 3, 1, 2)

            encoded_BB_tors = torch.ones([B, N_r, 4])
            encoded_BB_tors[..., 0] = torch.sin(BB_tors[..., 0])
            encoded_BB_tors[..., 1] = torch.cos(BB_tors[..., 0])
            encoded_BB_tors[..., 2] = torch.sin(BB_tors[..., 1])
            encoded_BB_tors[..., 3] = torch.cos(BB_tors[..., 1])

        else:
            geom_maps[..., 0] = (geom_maps[..., 0] / 10)
            geom_maps[..., 1] = geom_maps[..., 1] / math.pi
            geom_maps[..., 2] = geom_maps[..., 2] / math.pi
            geom_maps[..., 3] = (2 * geom_maps[..., 3] / math.pi) - 1

            # mask_gms = (geom_maps[0] == 1)
            # geom_maps[..., 1] = torch.where(mask_gms, 1, geom_maps[..., 1])
            # geom_maps[..., 2] = torch.where(mask_gms, 1, geom_maps[..., 2])
            # geom_maps[..., 3] = torch.where(mask_gms, 1, geom_maps[..., 3])
            
            encoded_geom_maps = geom_maps.permute(0, 3, 1, 2)

            BB_tors = torch.stack(list(BB_tors.values()), -1)
            encoded_BB_tors = ((BB_tors + math.pi)/(2 * math.pi))

        residue_index = torch.arange(N_r)


        representation = {"single": encoded_BB_tors.to(quataffine.device), "pair": encoded_geom_maps.to(quataffine.device), 
                          "residue_index": residue_index.to(quataffine.device), "affine": quataffine.to(quataffine.device)}

        return representation


    def affine2globalcrd(self, quataffine, scaler):

        assert len(quataffine.shape) == 3
        B, N_res, _ = quataffine.shape

        quaternion = quataffine[:, :, :4]
        translation = quataffine[:, :, 4:]

        quaternion = quaternion / (quaternion.square().sum(dim=-1, keepdims=True) + 1e-10).sqrt()

        rotation = rot_list_to_tensor(quat_to_rot(quaternion))

        rigid_group_bb_atom = rigid_group_bb_positions_mean[None, None :, :].repeat(B, N_res, 1, 1) / scaler
        rigid_group_bb_atom = rigid_group_bb_atom.to(quaternion.device)

        updated_rigid_group_bb_atom = torch.einsum("bnac,bndc->bnad", rigid_group_bb_atom, rotation) + translation[:, :, None, :]

        return updated_rigid_group_bb_atom

        # import pdb; pdb.set_trace()
        # save_pdb_file(translation[0].reshape(-1, 3).numpy() * self.config.structure_module.scale_factor * 2, \
        #     filename="/yrfs1/intern/yfliu25/protein_diffusion/check_protein.pdb")

        # return updated_rigid_group_bb_atom


    def apply_rot_to_vec_batch(self, rot, vec, unstack=False):

        if unstack:
            x, y, z = [vec[..., i] for i in range(3)]
        else:
            x, y, z = vec
        return [rot[0][0][:, :, None] * x + rot[0][1][:, :, None] * y + rot[0][2][:, :, None] * z,
                rot[1][0][:, :, None] * x + rot[1][1][:, :, None] * y + rot[1][2][:, :, None] * z,
                rot[2][0][:, :, None] * x + rot[2][1][:, :, None] * y + rot[2][2][:, :, None] * z]

