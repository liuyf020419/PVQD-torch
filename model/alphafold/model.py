import torch
from torch import nn
from torch.nn import functional as F

from .layers import *
from .evoformer import EmbeddingsAndEvoformer  
from .heads import *
from .folding import StructureModule

from .common import residue_constants

class AlphaFoldIteration(nn.Module):
    """A single recycling iteration of AlphaFold architecture.

    Computes ensembled (averaged) representations from the provided features.
    These representations are then passed to the various heads
    that have been requested by the configuration file. Each head also returns a
    loss which is combined as a weighted sum to produce the total loss.

    Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 3-22
    """
    def __init__(self, config, global_config, compute_loss=True):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.compute_loss = compute_loss

        self.evoformer = EmbeddingsAndEvoformer(
            config.embeddings_and_evoformer, global_config)

        self.masked_msa_head = MaskedMsaHead(config.heads.masked_msa, global_config, config.embeddings_and_evoformer.msa_channel)
        self.distogram_head = DistogramHead(config.heads.distogram, global_config, config.embeddings_and_evoformer.pair_channel)
        self.predicted_lddt_head = PredictedLDDTHead(config.heads.predicted_lddt, global_config, config.embeddings_and_evoformer.seq_channel)
        self.predicted_aligned_error_head = PredictedAlignedErrorHead(config.heads.predicted_aligned_error, global_config, config.embeddings_and_evoformer.pair_channel)
        self.experimentally_resolved_head = ExperimentallyResolvedHead(config.heads.experimentally_resolved, global_config, config.embeddings_and_evoformer.seq_channel)
        self.structure_module = StructureModule(
            config.heads.structure_module, global_config, 
            config.embeddings_and_evoformer.seq_channel,
            config.embeddings_and_evoformer.pair_channel,
            compute_loss
        )

        self.heads = {
            "masked_msa": "masked_msa_head",
            "distogram": "distogram_head",
            "structure_module": "structure_module",
            "experimentally_resolved": "experimentally_resolved_head",
            "predicted_lddt": "predicted_lddt_head",
            "predicted_aligned_error": "predicted_aligned_error_head",
        }

    def forward(self, 
                ensembled_batch,
                non_ensembled_batch,
                compute_loss=False,
                ensemble_representations=False,
                return_representations=False):
        num_ensemble = ensembled_batch['seq_length'].size(0)
        # import pdb;pdb.set_trace()
        # if not ensemble_representations:
        #     assert ensembled_batch['seq_length'].shape[0] == 1

        def slice_batch(i):
            b = {k: v[i] for k, v in ensembled_batch.items()}
            b.update(non_ensembled_batch)
            return b
        batch0 = slice_batch(0)
       
        representations = self.evoformer(batch0)

        # MSA representations are not ensembled so
        # we don't pass tensor into the loop.
        msa_representation = representations['msa']
        del representations['msa']

        if ensemble_representations:
            def body(x):
                """Add one element to the representations ensemble."""
                i, current_representations = x
                feats = slice_batch(i)
                representations_update = self.evoformer(feats)

                new_representations = {}
                for k in current_representations:
                    new_representations[k] = (
                        current_representations[k] + representations_update[k])
                return i+1, new_representations

            for ensemble_id in range(1, num_ensemble):
                ensemble_id, representations = body((ensemble_id, representations))

            for k in representations:
                if k != 'msa':
                    representations[k] /= num_ensemble.type_as(representations[k])
        
        representations['msa'] = msa_representation

        # compute_loss = True
        # import numpy as np
        # representations_raw = np.load('../alphafold_train/representations.npy', allow_pickle=True).item()
        # representations = pax.tree_map(lambda x: torch.from_numpy(x).to(batch0['aatype'].device), representations_raw)

        batch = batch0  # We are not ensembled from here on.

        total_loss = 0.
        ret = {}
        ret['representations'] = representations

        def loss(module, head_config, ret, name, filter_ret=True):
            if filter_ret:
                value = ret[name]
            else:
                value = ret
            loss_output = module.loss(value, batch)
            ret[name].update(loss_output)
            loss = head_config.weight * ret[name]['loss']
            return loss

        for name, head_name in self.heads.items():
            if name in ('predicted_lddt', 'predicted_aligned_error'):
                continue
            else:
                module = getattr(self, head_name)
                ret[name] = module(representations, batch)
                if 'representations' in ret[name]:
                    representations.update(ret[name].pop('representations'))
            if compute_loss:
                total_loss = total_loss + loss(module, self.config.heads.get(name), ret, name)

        if self.config.heads.get('predicted_lddt.weight', 0.0):
            ret['predicted_lddt'] = self.predicted_lddt_head(representations, batch)
            if compute_loss:
                total_loss = total_loss + loss(self.predicted_lddt_head, self.config.heads.predicted_lddt, ret, 'predicted_lddt', filter_ret=False)

        if ('predicted_aligned_error' in self.config.heads
            and self.config.heads.get('predicted_aligned_error.weight', 0.0)):
            ret['predicted_aligned_error'] = self.predicted_aligned_error_head(representations, batch)

            if compute_loss:
                total_loss = total_loss + loss(self.predicted_aligned_error_head, self.config.heads.predicted_aligned_error, ret, 'predicted_aligned_error', filter_ret=False)

        if compute_loss:
            return ret, total_loss
        else:
            return ret


class AlphaFold(nn.Module):
    def __init__(self, config, global_config, compute_loss=True):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.compute_loss = compute_loss

        self.alphafold_iteration = AlphaFoldIteration(config, global_config, compute_loss=True)
    
    def forward(self, 
                batch,
                ensemble_representations=False,
                return_representations=False):
        batch_size, num_residues = batch['aatype'].shape
      
        def get_prev(ret):
            new_prev = {
                'prev_pos':
                    ret['structure_module']['final_atom_positions'].detach(),
                'prev_msa_first_row': ret['representations']['msa_first_row'].detach(),
                'prev_pair': ret['representations']['pair'].detach(),
            }
            return new_prev

        def do_call(prev, recycle_idx, compute_loss):
            
            if self.config.resample_msa_in_recycling:
                num_ensemble = 1 #batch_size // (self.config.num_recycle + 1)
            
                # dynamic_slice_in_dim
                ensembled_batch = {}
                slice_start = recycle_idx * num_ensemble
                slice_size = num_ensemble
                for batch_key, batch_val in batch.items():
                    ensembled_batch[batch_key] = batch_val[slice_start: slice_start+slice_size]
            else:
                num_ensemble = batch_size
                ensembled_batch = batch
            # import pdb;pdb.set_trace()
            non_ensembled_batch = prev #pax.tree_map(lambda x: x, prev)
            return self.alphafold_iteration(
                ensembled_batch = ensembled_batch,
                non_ensembled_batch = non_ensembled_batch,
                ensemble_representations = ensemble_representations,
                compute_loss = compute_loss,
            )
        
        if self.config.num_recycle:
            emb_config = self.config.embeddings_and_evoformer
           
            prev = {
                'prev_pos': torch.zeros((num_residues, residue_constants.atom_type_num, 3)).to(batch['seq_mask']),
                'prev_msa_first_row': torch.zeros((num_residues, emb_config.msa_channel)).to(batch['seq_mask']),
                'prev_pair': torch.zeros((num_residues, num_residues, emb_config.pair_channel)).to(batch['seq_mask']),
            }

            if 'num_iter_recycling' in batch:
                # Training time: num_iter_recycling is in batch.
                # The value for each ensemble batch is the same, so arbitrarily taking
                # 0-th.
                num_iter = batch['num_iter_recycling'][0]

                # Add insurance that we will not run more
                # recyclings than the model is configured to run.
                num_iter = min(num_iter, self.config.num_recycle)
            else:
                # Eval mode or tests: use the maximum number of iterations.
                num_iter = self.config.num_recycle
            num_iter = num_iter -1
            for recycle_idx in range(num_iter):
                prev = get_prev(do_call(prev, recycle_idx=recycle_idx, compute_loss=False))
        else:
            prev = {}
            num_iter = 0
        ret = do_call(prev=prev, recycle_idx=num_iter, compute_loss=True)
        if self.compute_loss:
            ret = ret[0], [ret[1]]

        if not return_representations:
            del (ret[0] if self.compute_loss else ret)['representations']  # pytype: disable=unsupported-operands
        return ret
