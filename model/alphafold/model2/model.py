import torch
from torch import nn
from torch.nn import functional as F
import random
from .layers import *
from .evoformer import EmbeddingsAndEvoformer  
from .heads import *
from .folding import StructureModule
import logging

from alphafold.common import residue_constants

logger= logging.getLogger(__name__)


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

        masked_msa_head = MaskedMsaHead(config.heads.masked_msa, global_config, config.embeddings_and_evoformer.msa_channel)
        distogram_head = DistogramHead(config.heads.distogram, global_config, config.embeddings_and_evoformer.pair_channel)
        predicted_lddt_head = PredictedLDDTHead(config.heads.predicted_lddt, global_config, config.embeddings_and_evoformer.seq_channel)
        predicted_aligned_error_head = PredictedAlignedErrorHead(config.heads.predicted_aligned_error, global_config, config.embeddings_and_evoformer.pair_channel)
        experimentally_resolved_head = ExperimentallyResolvedHead(config.heads.experimentally_resolved, global_config, config.embeddings_and_evoformer.seq_channel)
        structure_module = StructureModule(
            config.heads.structure_module, global_config, 
            config.embeddings_and_evoformer.seq_channel,
            config.embeddings_and_evoformer.pair_channel,
            compute_loss
        )
        if self.config.heads.get('predicted_aligned_error.weight', 0.0):
            for p in predicted_aligned_error_head.parameters():
                p.requires_grad=False
        self.heads= nn.ModuleDict({
            "masked_msa": masked_msa_head,
            "distogram": distogram_head,
            "structure_module": structure_module,
            "experimentally_resolved": experimentally_resolved_head,
            "predicted_lddt": predicted_lddt_head,
            "predicted_aligned_error": predicted_aligned_error_head,
        })
        

    def forward(self, 
                ensembled_batch,
                non_ensembled_batch,
                compute_loss=False,
                ensemble_representations=False,
                return_representations=False):
        batch_size, num_ensemble = ensembled_batch['seq_length'].shape
        
        if not ensemble_representations:
            assert num_ensemble == 1, f"ensemble number is {num_ensemble}"


        def slice_batch(i):
            b = {k: v[:, i] for k, v in ensembled_batch.items()}
            b.update(non_ensembled_batch)
            return b
        def expand_beam(x:torch.Tensor, n:int):
            repeat_mot= (1,n) + (1,)*(x.dim()-1)
            x= x.unsqueeze(1).repeat(repeat_mot)
            x= flatten_prev_dims(x,2) 
        def flatten_batch():
            b= {k:flatten_prev_dims(v,2) for k,v in ensembled_batch.items()}
            for k,v in non_ensembled_batch:
                b[k] = expand_beam(v, num_ensemble)
        
        def split_ens(x:torch.Tensor, shape0, shape1):
            return x.veiw((shape0,shape1) + x.shape[1:])

        batch0 = slice_batch(0)
        if num_ensemble ==1:
            representations = self.evoformer(batch0)
        else:
            representations = self.evoformer(flatten_batch())
            representations = {k:split_ens(v, batch_size, num_ensemble) for k,v in representations.items()}
            msa= representations['msa'][:0]
            del representations['msa']
            representations = {k:v.mean(1) for k,v in  representations}
            representations['msa'] = msa
           
            # reprlist= [self.evoformer(slice_batch(i)) for i in range(num_ensemble)]
            # keys= list(reprlist[0].keys())
            # del keys['msa']
            # mean_repr= lambda k: sum([r[k] for r in reprlist])/num_ensemble
            # representations = {k:mean_repr(k) for k in keys}
            # representations['msa'] = reprlist[0]['msa']

        # compute_loss = True
        # import numpy as np
        # representations_raw = np.load('../alphafold_train/representations.npy', allow_pickle=True).item()
        # representations = pax.tree_map(lambda x: torch.from_numpy(x).to(batch0['aatype'].device), representations_raw)
        # import pdb;pdb.set_trace()
        batch = batch0  # We are not ensembled from here on.

        total_loss = 0.
        ret = {}
        ret['representations'] = representations

        if getattr(self.global_config, "extract_hidden", False):
            return ret

        def loss(module, head_config, ret, name, filter_ret=True):
            if filter_ret:
                value = ret[name]
            else:
                value = ret
            loss_output = module.loss(value, batch)
            ret[name].update(loss_output)
            if ret[name]['loss'].ndim >0:
                ret[name]['loss']= ret[name]['loss'].mean()
            loss = head_config.weight * ret[name]['loss']
            return loss

        for name, module in self.heads.items():
            
            if name in ('predicted_lddt', 'predicted_aligned_error'):
                continue
            else:
                ret[name] = module(representations, batch)
                if 'representations' in ret[name]:
                    representations.update(ret[name].pop('representations'))
            if compute_loss:
                sloss = loss(module, self.config.heads.get(name), ret, name)
                
                total_loss = total_loss + sloss
                
        if self.config.heads.get('predicted_lddt.weight', 0.0):
            ret['predicted_lddt'] = self.heads['predicted_lddt'](representations, batch)
            if compute_loss:
                sloss=loss(self.heads['predicted_lddt'], self.config.heads.predicted_lddt, ret, 'predicted_lddt', filter_ret=False)
                total_loss = total_loss + sloss

        if ('predicted_aligned_error' in self.config.heads
            and self.config.heads.get('predicted_aligned_error.weight', 0.0)):
            ret['predicted_aligned_error'] = self.heads["predicted_aligned_error"](representations, batch)

            if compute_loss:
                sloss=loss(self.heads["predicted_aligned_error"], self.config.heads.predicted_aligned_error, ret, 'predicted_aligned_error', filter_ret=False)
                total_loss = total_loss + sloss
        
        if compute_loss:
            return ret, total_loss
        else:
            return ret


class AlphaFold(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.config = config
        self.global_config = global_config
        # self.compute_loss = compute_loss

        self.alphafold_iteration = AlphaFoldIteration(config, global_config, compute_loss=True)
    
    def forward(self, 
                batch,
                ensemble_representations=False,
                recycle_num= None
                ):
        batch_size,iter_num, num_residues = batch['aatype'].shape
        def get_prev(ret):
            new_prev = {
                'prev_pos':
                    ret['structure_module']['final_atom_positions'].detach().clone(),
                'prev_msa_first_row': ret['representations']['msa_first_row'].detach().clone(),
                'prev_pair': ret['representations']['pair'].detach().clone(),
            }
            ret = None
            return new_prev
        
        num_ensemble= self.config.num_ensemble
        # for batch processing we should keep same iter_num in data(max_recycle_num*num_ensemble), and adjust recyle num here
        assert num_ensemble*self.config.max_num_recycle <=iter_num,\
            f"iter_num in data is {iter_num}, while num_ensemble={num_ensemble}, max_num_recycle={self.config.max_num_recycle}"
        if not self.training:
            num_recycle= self.config.max_num_recycle
        else:
            num_recycle = random.randint(1, self.config.max_num_recycle)
        if recycle_num is not None:
            num_recycle = recycle_num    
            
        emb_config = self.config.embeddings_and_evoformer
        prev = {
            'prev_pos': torch.zeros((batch_size,num_residues, residue_constants.atom_type_num, 3)).to(batch['seq_mask']),
            'prev_msa_first_row': torch.zeros((batch_size,num_residues, emb_config.msa_channel)).to(batch['seq_mask']),
            'prev_pair': torch.zeros((batch_size,num_residues, num_residues, emb_config.pair_channel)).to(batch['seq_mask']),
        }
        def do_call(prev, recycle_idx, compute_loss):
            ensembled_batch = {
                k:v[:, recycle_idx*num_ensemble:(recycle_idx+1)*num_ensemble]
                for k,v in batch.items()
            }
            return self.alphafold_iteration(
                ensembled_batch = ensembled_batch,
                non_ensembled_batch = prev,
                ensemble_representations = ensemble_representations,
                compute_loss = compute_loss,
            )
        
        with torch.no_grad():
            for i in range(num_recycle-1):
                iter_out = do_call(prev, i, compute_loss=False)
                prev= get_prev(iter_out)
                if self.global_config.is_inference:
                    torch.cuda.empty_cache()
                    logger.debug(f'recycle {i} done')
                    del iter_out

        if not self.global_config.is_inference:
            ret, total_loss = do_call(prev=prev, recycle_idx=num_recycle -1, compute_loss=True)
            return ret, total_loss
        else:
            ret = do_call(prev=prev, recycle_idx=num_recycle -1, compute_loss=False)
            return ret
