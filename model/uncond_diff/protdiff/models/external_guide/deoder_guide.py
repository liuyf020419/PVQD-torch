import sys,os
import logging

import numpy as np
import torch

sys.path.append('/raw7/superbrain/yfliu25/VQstructure/vqgvp_rvq_multichain/protdiff')
from models.vqstructure import VQStructure


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger= logging.getLogger(__name__)


def load_checkpoint(checkpoint_path, model):
    last_cp= checkpoint_path
    if not os.path.exists(last_cp):
        logger.error(f'checkpoint file {last_cp} not exist, ignore load_checkpoint')
        return
    with open(last_cp,'rb') as f:
        logger.info(f'load checkpoint: {checkpoint_path}')
        state = torch.load(f, map_location=torch.device("cpu"))
    model.load_state_dict(state['model'])
    return model, state['update_num']


class VQDecoderGuide(object):
    def __init__(self, config, pretrained_ckpt) -> None:
        self.config = config
        self.vqdecoder = VQStructure(
            self.config.vqstucture_model_config, 
            self.config.vqstucture_global_config
            )
        self.load_pretrained(pretrained_ckpt)
        self.vqdecoder.eval()
        self.vqdecoder.cuda()

        
    def load_pretrained(self, pretrained_ckpt):
        _, step = load_checkpoint(pretrained_ckpt, self.vqdecoder)


    def decoder_forward(self, pred_hidden, batch):
        pred_latent = pred_hidden

        min_codebook_latent, min_pred_indices, _ = \
            self.vqdecoder.codebook.compute_each_codebook(
                self.vqdecoder.codebook.codebook_layer[0], pred_latent)
        min_codebook_input = self.vqdecoder.codebook.post_quant(min_codebook_latent)

        reps = self.vqdecoder.decode(
            min_codebook_input, batch['single_mask'][0][None], \
                batch['single_res_rel'][0][None], batch['chain_idx'][0][None], batch['entity_idx'][0][None],\
                batch['pair_res_idx'][0][None], batch['pair_chain_idx'][0][None], batch['pair_same_entity'][0][None])
        affine_p, single_rep, pair_rep_act = reps

        return affine_p
        ca_coords = affine_p[..., 4:] # B, L, 3
        return ca_coords
