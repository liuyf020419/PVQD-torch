import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

import numpy as np

from .encoder_module.gvpstructure_embedding import GVPStructureEmbedding
from .encoder_module.protein_mpnn_embedding import ProteinMPNNEmbedding

from .codebook import Codebook, ResidualCodebook

from .stack_attention import \
    TransformerStackDecoder, EvoformerStackDecoder, TransformerRotary, SingleToPairModule, IPAattentionStackedDecoder
from .folding_af2.ipa_rigid_net import StructureModule
from .folding_af2 import r3

from .nn_utils import TransformerPositionEncoding, \
    make_low_resolution_mask, generate_new_affine, \
        fape_loss, make_mask, mask_loss, make_pairidx_from_singleidx, \
            distogram_loss, aatype_ce_loss, downsampling_single_idx, downsampling_pair_idx, \
                get_batch_quataffine, fape_loss_multichain

from .protein_utils.write_pdb import write_multichain_from_atoms, fasta_writer
from .protein_utils.add_o_atoms import add_atom_O, batch_add_atom_O_new
from .protein_utils.rigid import affine_to_frame12, affine_to_pos
from .protein_utils.covalent_loss import structural_violation_loss
from .protein_utils.symmetry_loss import center_mass_loss_batch

af2_restypes = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V'
]
af2_aatype_to_index = {restype: i for i, restype in enumerate(af2_restypes)}
af2_index_to_aatype = {v:k for k, v in af2_aatype_to_index.items()}


def loss_dict_fp32(dict:dict):
    fp32_dict = {}
    for k, v in dict.items():
        fp32_dict[k] = v.float()

    return fp32_dict


class VQStructure(nn.Module):
    def __init__(self, config, global_config):
        super(VQStructure, self).__init__()
        self.config = config
        self.global_config = global_config
        self.down_sampling_scale = self.global_config.down_sampling_scale

        ### feature extraction
        if self.global_config.feature_encoder_type == 'GVPEncoder':
            self.encoder = GVPStructureEmbedding(
                self.config.gvp_embedding, self.global_config, self.down_sampling_scale)
        elif self.global_config.feature_encoder_type == 'ProteinMPNN':
            self.encoder = ProteinMPNNEmbedding(
                self.config.protein_mpnn_embedding, self.global_config, self.down_sampling_scale)
        else:
            raise ValueError(f'{self.global_config.feature_encoder_type} unknown')

        ### feature transform eg. transformerrotary
        self.stacked_encoder = self.make_stackedtransformer(
            self.global_config.low_resolution_encoder_type, 
            self.config, single_in_dim=self.encoder.embed_dim, 
            layer_num=self.global_config.encoder_layer)

        ### binary codebook
        if not self.global_config.residualvq:
            self.codebook = Codebook(
                self.config.codebook, self.encoder.embed_dim, 
                head_num=self.config.codebook.head_num, 
                seperate_codebook_per_head=self.config.codebook.seperate_codebook_per_head)

        else:
            self.codebook = ResidualCodebook(
                self.config.residual_codebook, self.encoder.embed_dim, 
                codebook_num=self.config.residual_codebook.codebook_num,
                shared_codebook=self.config.residual_codebook.shared_codebook,
                codebook_dropout=self.config.residual_codebook.codebook_dropout)

        ### decoder
        # self.position_embedding = TransformerPositionEncoding(
        #     global_config.max_seq_len, self.encoder.embed_dim)
        self.single_res_embedding = TransformerPositionEncoding(
            global_config.max_seq_len, self.encoder.embed_dim)
        self.single_chain_embedding = nn.Embedding(
            global_config.max_chain_len, self.encoder.embed_dim)
        self.single_entity_embedding = nn.Embedding(
            global_config.max_entity_len, self.encoder.embed_dim)

        self.high_resolution_decoder = self.make_stackedtransformer(
            self.global_config.high_resolution_decoder_type, 
            self.config, self.encoder.embed_dim, self.global_config.single_rep_dim,
            layer_num=self.global_config.high_resolution_decoder_layer)
        self.decoder = self.high_resolution_decoder

        if self.global_config.high_resolution_decoder_type != 'IPAattention':
            self.structure_module = StructureModule(
                self.config.structure_module, self.global_config, self.global_config.single_rep_dim)

            if self.global_config.high_resolution_decoder_type == 'Evoformer':
                self.pair_act = nn.Linear(
                    self.high_resolution_decoder.pair_channel, 
                    self.config.structure_module.pair_channel)
            else:
                self.pair_act = SingleToPairModule(
                    self.config.structure_module, self.global_config, 
                    self.high_resolution_decoder.single_channel,
                    self.config.structure_module.pair_channel)
            distogram_in_ch = self.config.structure_module.pair_channel
            aatype_in_ch = self.global_config.single_rep_dim
        else:
            distogram_in_ch = self.decoder.pair_channel
            aatype_in_ch = self.global_config.single_rep_dim

        self.distogram_predictor = self.build_distogram_predictor(distogram_in_ch)
        if self.global_config.loss_weight.aatype_celoss > 0.0:
            self.aatype_predictor = self.build_aatype_predictor(aatype_in_ch)


    def make_stackedtransformer(self, decoder_type, config, single_in_dim, out_dim=None, layer_num=None):
        if decoder_type == 'TransformerWithBias':
            stacked_decoder = TransformerStackDecoder(
                config.transformerwithbias, self.global_config, 
                with_bias=True, single_in_dim=single_in_dim, 
                out_dim =out_dim, layer_num=layer_num
                )
        if decoder_type == 'TransformerRotary':
            stacked_decoder = TransformerRotary(
                config.transformerRotary, self.global_config, 
                single_in_dim, out_dim, layer_num= layer_num
                )
        elif decoder_type == 'Evoformer':
            stacked_decoder = EvoformerStackDecoder(
                config.evoformer, self.global_config, single_in_dim, out_dim=out_dim)
        elif decoder_type == 'IPAattention':
            stacked_decoder = IPAattentionStackedDecoder(
                config.ipa_attention, self.global_config, single_in_dim, out_dim=out_dim)
        else:
            raise ValueError(f'{stacked_decoder} unknown')

        return stacked_decoder


    def build_distogram_predictor(self, pair_channel):
        out_num = self.config.distogram_pred.distogram_args[-1]
        distogram_predictor = nn.Sequential(
            nn.Linear(pair_channel, out_num),
            nn.ReLU(),
            nn.Linear(out_num, out_num))
        
        return distogram_predictor


    def build_aatype_predictor(self, single_channel):
        if self.global_config.additional_aatype_decoder_in_codebook:
            aatype_ce_head = TransformerRotary(
                self.config.aatype_transformerRotary, 
                self.global_config, single_channel,20)
        else:
            aatype_ce_head = nn.Sequential(
                nn.Linear(single_channel, single_channel),
                nn.ReLU(),
                nn.LayerNorm(single_channel),
                nn.Linear(single_channel, 20))

        return aatype_ce_head


    def forward(self, batch, return_all=False, use_codebook_num=4):
        device = batch['aatype'].device
        dtype = batch['gt_pos'].dtype
        batchsize, L, N, _ = batch['gt_pos'].shape
        make_mask(batch['len'], batchsize, L, batch, dtype)
        
        codebook_mapping, codebook_indices, q_loss, q_loss_dict, encoded_feature, z_q_out = self.encode(batch, return_all, use_codebook_num)
        if self.global_config.high_resolution_decoder_type != 'IPAattention':
            reps = self.decode( 
                codebook_mapping, batch['single_mask'], batch['single_res_rel'], batch['pair_res_rel'])
            if self.global_config.high_resolution_decoder_type == 'Evoformer':
                single_rep, pair_rep = reps
                pair_rep_act = self.pair_act(pair_rep)
            else:
                single_rep = reps
                pair_rep_act = self.pair_act(single_rep)

            zero_affine = generate_new_affine(batch['single_mask'])
            representations = {
                "single": single_rep,
                "pair": pair_rep_act,
                "frame": zero_affine,
                "single_mask": batch["single_mask"],
                "pair_mask": batch["pair_mask"]
                }
            representations = {k: v.float() for k, v in representations.items()}
            pred_dict = self.structure_module(representations=representations)
            affine_p = pred_dict['traj']
        else:
            reps = self.decode(
                codebook_mapping, batch['single_mask'], \
                    batch['single_res_rel'], batch['chain_idx'], batch['entity_idx'],\
                    batch['pair_res_idx'], batch['pair_chain_idx'], batch['pair_same_entity'])
            affine_p, single_rep, pair_rep_act = reps
            affine_p = affine_p[None]
            representations = {
                "single": single_rep,
                "pair": pair_rep_act
                }

        ## distogram loss
        pred_distogram = self.distogram_predictor(representations['pair'])
        dist_loss = distogram_loss(
            batch['gt_pos'], pred_distogram, self.config.distogram_pred, batch['pair_mask'])
        ## aa loss
        if self.global_config.loss_weight.aatype_celoss > 0.0:
            if self.global_config.decode_aatype_in_codebook:
                if self.global_config.additional_aatype_decoder_in_codebook:
                    pred_aatype = self.aatype_predictor(
                        codebook_mapping, batch['single_res_rel'], batch['single_mask'], batch=batch).float()
                else:
                    pred_aatype = self.aatype_predictor(codebook_mapping.float())
            elif self.global_config.decode_aatype_in_deocoder:
                pred_aatype = self.aatype_predictor(representations['single'])
            else:
                pred_aatype = self.aatype_predictor(encoded_feature.float())
            if self.training:
                if batch['aatype_mask'] is not None:
                    aa_mask = batch['aatype_mask']
                else:
                    aa_mask = batch['single_mask']
            else:
                aa_mask = batch['single_mask']

            aatype_loss, aatype_acc = aatype_ce_loss(batch['aatype'], pred_aatype, aa_mask)
        ## aa loss
        affine_0 = r3.rigids_to_quataffine_m(r3.rigids_from_tensor_flat12(batch['gt_backbone_frame'].float())).to_tensor()[..., 0, :]

        losses_dict, fape_dict = fape_loss_multichain(
            affine_p, affine_0, batch['gt_pos'][..., :3, :], batch['single_mask'], batch['chain_idx'], self.global_config.fape)

        # q_loss, dist_loss, ceaatype loss in dict
        atom14_positions = fape_dict['coord'][-1]
        losses_dict.update({'q_loss': q_loss})
        losses_dict.update(q_loss_dict)
        losses_dict.update({'ditogram_classify_loss': dist_loss})


        if self.global_config.loss_weight.aatype_celoss > 0.0:
            losses_dict.update({'aatype_celoss': aatype_loss})
            losses_dict.update({'aatype_acc': aatype_acc})
        if self.global_config.loss_weight.violation_loss > 0.0:
            violation_loss = structural_violation_loss(
                batch, atom14_positions, self.global_config.violation_config)
            losses_dict.update({'violation_loss': violation_loss})
        if self.global_config.loss_weight.mass_center_loss > 0.0:
            mass_center_loss = center_mass_loss_batch(
                atom14_positions, batch['gt_pos'][..., :3, :], 
                batch['single_mask'], batch['chain_idx'])
            losses_dict.update({'mass_center_loss': mass_center_loss})
            # import pdb; pdb.set_trace()
        
        losses_dict = mask_loss(batch['loss_mask'], losses_dict)
        loss = sum([losses_dict[k].mean() * 
                        self.global_config.loss_weight[k] \
                for k in self.global_config.loss_weight if k in losses_dict.keys()])
        losses_dict['loss'] = loss
        
        losses_dict = loss_dict_fp32(losses_dict)
        fape_dict = loss_dict_fp32(fape_dict)

        if return_all:
            all_dict = {
                "codebook_mapping": z_q_out,
                "codebook_indices": codebook_indices,
                "decoder_rep": representations,
                "pred_aatype": pred_aatype,
                "affine_p": affine_p,
                "coords_dict": fape_dict,
                "loss": losses_dict
            }
            return all_dict
        else:
            return losses_dict, codebook_indices


    def encode(self, batch, return_all_indices=False,use_codebook_num=4):
        encoded_feature = self.encoder(batch)
        encoder_single_mask = single_mask = batch['single_mask']
        batchsize, res_num, _ = encoded_feature.shape
        dtype = encoded_feature.dtype
        device = batch['aatype'].device

        if self.down_sampling_scale > 1:
            encoder_single_mask = make_low_resolution_mask(single_mask, self.down_sampling_scale).to(dtype)
        if (batch.__contains__('chain_idx') and batch.__contains__('encode_split_chain')):
            between_chain_mask = torch.where((batch['chain_idx'][:, None] - batch['chain_idx'][:, :, None]) == 0, 0., 1.)
            encode_split_chain = batch['encode_split_chain'][:, None, None] # 1. means split (mask), 0. means merge (visiable)
            pair_mask = between_chain_mask * encode_split_chain # 0. means visiable, 1. means mask
            pair_mask = 1. - pair_mask # 1. means visiable, 0. means mask
            encoded_feature = self.stacked_encoder(
                encoded_feature, batch['single_res_rel'], single_mask=encoder_single_mask, pair_mask=pair_mask)
        else:
            encoded_feature = self.stacked_encoder(encoded_feature, single_mask=encoder_single_mask)
        codebook_mapping, codebook_indices, q_loss, q_loss_dict, stacked_z_q, z_q_out = self.codebook(encoded_feature, return_all_indices, use_codebook_num)

        q_loss_reduced = q_loss * encoder_single_mask
        q_loss_reduced = torch.sum(q_loss_reduced) / (torch.sum(encoder_single_mask) + 1e-6)
        # import pdb; pdb.set_trace()
        # if len(codebook_indices) == 1:
        #     q_loss_dict['min_indices_num_count'] = torch.tensor( [len(Counter(codebook_indices[0].detach().tolist()))] ).float().to(device)/batchsize
        q_loss_dict = {k: torch.sum(v * encoder_single_mask) / (torch.sum(encoder_single_mask) + 1e-6)\
                for k, v in q_loss_dict.items()}

        if self.codebook.head_num == 2:
            q_loss_dict['min_indices_num_1_count'] = torch.tensor( [len(Counter(codebook_indices[0].detach().tolist()))] ).float().to(device)/batchsize
            q_loss_dict['min_indices_num_2_count'] = torch.tensor( [len(Counter(codebook_indices[1].detach().tolist()))] ).float().to(device)/batchsize
        
        else:
            q_loss_dict['min_indices_num_count'] = torch.tensor( [len(Counter(codebook_indices[0].detach().tolist()))] ).float().to(device)/batchsize
            
        return codebook_mapping, codebook_indices, q_loss_reduced.float(), q_loss_dict, encoded_feature, z_q_out


    def decode(self, single, single_mask, single_idx=None, chain_idx=None, entity_idx=None, pair_res_idx=None, pair_chain_idx=None, pair_same_entity=None):
        dtype = single.dtype
        device = single.device
        batchsize, res_num = single_mask.shape
        if self.down_sampling_scale >1:
            single = F.interpolate(
                single.transpose(-1, -2), 
                scale_factor=self.down_sampling_scale).transpose(-1, -2)[:, :res_num]
      
        if self.global_config.high_resolution_decoder_type != 'TransformerRotary':
            padding_mask = ~single_mask.bool()
            if single_idx is None:
                res_idx = torch.arange(res_num).to(device)
                pair_res_idx = make_pairidx_from_singleidx(res_idx, self.global_config.pair_res_range)[None, ...]
                single_idx = res_idx[None, :] * ~padding_mask + self.global_config.pad_num*padding_mask
            else:
                single_idx = single_idx * ~padding_mask + self.global_config.pad_num*padding_mask
                chain_idx = (chain_idx * ~padding_mask).long() + self.global_config.pad_chain_num*padding_mask
                entity_idx = (entity_idx * ~padding_mask).long() + self.global_config.pad_entity_num*padding_mask
            
            # single = single + self.position_embedding(single_idx, index_select=True).to(dtype)
            single = single + self.single_res_embedding(single_idx, index_select=True).to(dtype) + \
                self.single_chain_embedding(chain_idx).to(dtype) + \
                    self.single_entity_embedding(entity_idx).to(dtype)

        if self.global_config.high_resolution_decoder_type == 'Evoformer':
            high_resolution_single = self.decoder(
                single, single_mask, pair_res_idx)
        elif self.global_config.high_resolution_decoder_type == 'TransformerRotary':
            high_resolution_single = self.decoder(
                single, single_mask)
        elif self.global_config.high_resolution_decoder_type == 'IPAattention':
            high_resolution_single = self.decoder(
                # single.float(), single_mask.float(), pair_res_idx, pair_chain_idx, pair_same_entity)
                single, single_mask, pair_res_idx, pair_chain_idx, pair_same_entity)
        else:
            high_resolution_single = self.decoder(
                single, single_mask, pair_res_idx)

        return high_resolution_single


    def calculate_lambda(self, perceptual_loss, gan_loss):
        last_layer = self.decoder[-1].model[-1]
        last_layer_weight = last_layer.weight
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]

        λ = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        λ = torch.clamp(λ, 0, 1e4).detach()
        return 0.8 * λ


    @torch.no_grad()
    def sampling(self, batch, pdb_prefix, return_all=True, save_rep=False, verbose_indices=True, compute_sc_identity=True):
        if return_all:
            reduced_chain_idx = list(set(batch['chain_idx'][0].tolist()))
            traj_pos = []
            for chain_label in reduced_chain_idx:
                traj_pos.append(
                    add_atom_O(
                        batch['traj_pos'][0][batch['chain_idx'][0] == chain_label].detach().cpu().numpy()[..., :3, :]).reshape(-1, 3)
                    )
            write_multichain_from_atoms(traj_pos, 
                    f'{pdb_prefix}_vqrecon_traj.pdb', natom=4)
            # traj_pos = add_atom_O(batch['traj_pos'].detach().cpu().numpy()[0, :, :3, :])
            # write_multichain_from_atoms([traj_pos.reshape(-1, 3)], 
            #         f'{pdb_prefix}_vqrecon_traj.pdb', natom=4)
            gt_coord4 = []
            for chain_label in reduced_chain_idx:
                gt_coord4.append(
                    add_atom_O(
                        batch['gt_pos'][0][batch['chain_idx'][0] == chain_label].detach().cpu().numpy()[..., :3, :]).reshape(-1, 3)
                    )
            write_multichain_from_atoms(gt_coord4, 
                    f'{pdb_prefix}_vqrecon_gt.pdb', natom=4)
            # gt_coord4 = add_atom_O(batch['gt_pos'].detach().cpu().numpy()[0, :, :3, :])
            # write_multichain_from_atoms([gt_coord4.reshape(-1, 3)], 
            #         f'{pdb_prefix}_vqrecon_gt.pdb', natom=4)

        all_rep_dict = self(batch, return_all=True)
        batchsize, res_num = all_rep_dict['pred_aatype'].shape[:2]
        coords_dict = all_rep_dict['coords_dict']
        pred_aatype = all_rep_dict['pred_aatype']
        codebook_num = len(all_rep_dict['codebook_indices'])
        if return_all:
            gt_aatype_af2idx = batch['aatype'][0].detach().cpu().numpy().tolist()
            gt_aatype_str = ''.join([af2_index_to_aatype[aa] for aa in gt_aatype_af2idx])
            fasta_dict = {'native_seq': gt_aatype_str}

            # for debug
            if all_rep_dict.__contains__('loss'):
                losses_dict = all_rep_dict['loss']
                intra_fape = losses_dict['intra_unclamp_fape_loss'].item()
                intra_clamp_fape = losses_dict['intra_clamp_fape_loss'].item()
                inter_fape = losses_dict['inter_unclamp_fape_loss'].item()
                inter_clamp_fape = losses_dict['inter_clamp_fape_loss'].item()
                print(f'intra-fape loss: {round(intra_fape, 3)}; intra-clamp fape: {round(intra_clamp_fape, 3)}')
                print(f'inter-fape loss: {round(inter_fape, 3)}; inter-clamp fape: {round(inter_clamp_fape, 3)}')

            if verbose_indices:
                if len(all_rep_dict['codebook_indices']) == 2:
                    indices_0_counter = Counter(all_rep_dict['codebook_indices'][0].tolist())
                    indices_1_counter = Counter(all_rep_dict['codebook_indices'][1].tolist())
                    indices_0_used = len(indices_0_counter)
                    indices_1_used = len(indices_1_counter)
                    mostcommon_10_used_indices_0 = indices_0_counter.most_common(10)
                    mostcommon_10_used_indices_1 = indices_1_counter.most_common(10)
                    print(f'codebook0: {round(indices_0_used/res_num, 3)} used; codebook1: {round(indices_1_used/res_num, 3)} used;')
                    print(f'codebook0 10 mostcommon: {mostcommon_10_used_indices_0}')
                    print(f'codebook1 10 mostcommon: {mostcommon_10_used_indices_1}')

            if compute_sc_identity:
                batch['traj_pos'] = coords_dict['coord'][-1]
                batch['traj_backbone_frame'] = get_batch_quataffine(coords_dict['coord'][-1])
                recycle_all_rep_dict = self(batch, return_all=True)
                recycle_coords_dict = recycle_all_rep_dict['coords_dict']
                recycle_losses_dict = recycle_all_rep_dict['loss']

                sc_ident_list = []
                for cb_idx in range(len(all_rep_dict['codebook_indices'])):
                    # import pdb; pdb.set_trace()
                    cb_indices = all_rep_dict['codebook_indices'][cb_idx]
                    recycle_cb_indices = recycle_all_rep_dict['codebook_indices'][cb_idx]
                    sc_ident_list.append(round(((cb_indices == recycle_cb_indices).sum()/len(recycle_cb_indices)).item(), 3))

                # sc_clamp_fape = recycle_losses_dict['clamp_fape_loss'].item()
                sc_intra_fape = recycle_losses_dict['intra_unclamp_fape_loss'].item()
                sc_intra_clamp_fape = recycle_losses_dict['intra_clamp_fape_loss'].item()
                sc_inter_fape = recycle_losses_dict['inter_unclamp_fape_loss'].item()
                sc_inter_clamp_fape = recycle_losses_dict['inter_clamp_fape_loss'].item()

                all_rep_dict['sc identity'] = sc_ident_list
                all_rep_dict['sc intra_loss'] = sc_intra_clamp_fape
                all_rep_dict['sc inter_loss'] = sc_inter_clamp_fape
                print(f'codebook sc identity: {sc_ident_list};')
                print(f'sc intra-fape loss: {round(sc_intra_fape, 3)}; sc intra-clamp fape: {round(sc_intra_clamp_fape, 3)}')
                print(f'sc inter-fape loss: {round(sc_inter_fape, 3)}; sc inter-clamp fape: {round(sc_inter_clamp_fape, 3)}')

            for batch_idx in range(batchsize):
                traj_coord_0 = []
                for chain_label in reduced_chain_idx:
                    traj_coord_0.append(
                        add_atom_O(
                            coords_dict['coord'][-1, batch_idx][batch['chain_idx'][0] == chain_label].detach().cpu().numpy()[..., :3, :]).reshape(-1, 3)
                        )
                write_multichain_from_atoms(traj_coord_0, 
                        f'{pdb_prefix}_vqrecon_batch_{batch_idx}.pdb', natom=4)
                # traj_coord_0 = add_atom_O(coords_dict['coord'][-1, batch_idx].detach().cpu().numpy()[..., :3, :])
                # write_multichain_from_atoms([traj_coord_0.reshape(-1, 3)], f'{pdb_prefix}_vqrecon_batch_{batch_idx}.pdb', natom=4)
                if compute_sc_identity:
                    recycle_traj_coord_0 = []
                    for chain_label in reduced_chain_idx:
                        recycle_traj_coord_0.append(
                            add_atom_O(
                                recycle_coords_dict['coord'][-1, batch_idx][batch['chain_idx'][0] == chain_label].detach().cpu().numpy()[..., :3, :]).reshape(-1, 3)
                            )
                    write_multichain_from_atoms(recycle_traj_coord_0, 
                            f'{pdb_prefix}_recycle_vqrecon_batch_{batch_idx}.pdb', natom=4)
                    # recycle_traj_coord_0 = add_atom_O(recycle_coords_dict['coord'][-1, batch_idx].detach().cpu().numpy()[..., :3, :])
                    # write_multichain_from_atoms([recycle_traj_coord_0.reshape(-1, 3)], f'{pdb_prefix}_recycle_vqrecon_batch_{batch_idx}.pdb', natom=4)

                pred_aatype_logits = torch.argmax(pred_aatype[batch_idx], -1).reshape((-1, )).detach().cpu().numpy()
                pred_aatype_str = ''.join([af2_index_to_aatype[aa] for aa in pred_aatype_logits])

            fasta_dict.update({f'predicted_{batch_idx}': pred_aatype_str})
            fasta_writer(fasta_f=f'{pdb_prefix}_vqrecon.fasta', fasta_dict=fasta_dict)

        if save_rep:
            np.save(f'{pdb_prefix}_vqstructure_rep.npy', all_rep_dict)
            ident = (np.array(list(pred_aatype_str)) == np.array(list(gt_aatype_str))).sum() / len(gt_aatype_str)
            return {
                'intra_fape': intra_fape, 'intra_clamp_fape': intra_clamp_fape, 
                'inter_fape': inter_fape, 'inter_clamp_fape': inter_clamp_fape, 'ident': ident
                }

        else:
            reshaped_indices = torch.stack([all_rep_dict['codebook_indices'][cb_idx].reshape(batchsize, res_num)\
                 for cb_idx in range(codebook_num)]).permute(1, 0, 2) # B, C, N

            return reshaped_indices


            
                