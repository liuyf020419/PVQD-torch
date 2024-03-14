import contextlib
import logging
import sys
from collections import OrderedDict
import time
import sys,os, json
import argparse
import numpy as np

from typing import Any, Dict, List
from ml_collections import ConfigDict
import torch
import torch.nn as nn


from protdiff.models.vqstructure import VQStructure

def load_config(path)->ConfigDict:
    return ConfigDict(json.loads(open(path).read()))


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger= logging.getLogger(__name__)


def build_parser():
    parser = argparse.ArgumentParser(description='Alphafold2')
    parser.add_argument('--gen_dir', type=str, default=None, help='generate dir')
    parser.add_argument('--model_path', type=str, default=None, help='path to checkpoint file')
    parser.add_argument('--fixed_model_path', type=str, default=None, help='path to fixed checkpoint file')
    parser.add_argument('--root_dir', type=str, default=None, help='project path')

    parser.add_argument('--gen_tag', type=str, default='', help='gen_tag')
    parser.add_argument('--seed', type=int, default=1234, help='Seed for PyTorch random number generators')
    parser.add_argument('--sample_from_raw_pdbfile', action='store_true', help='sample from raw pdbfile or processed file')
    parser.add_argument('--max_sample_num', type=int, default=10000000, help='maximum number of samples for testing or application')
    
    parser.add_argument('--write_pdbfile', action='store_true', help='write pdb file when testing model')
    parser.add_argument('--indices_f', type=str, default=None, help='pdb root for application')
    parser.add_argument('--batch_size', type=int, default=1, help='batchsize for each protein')

    return parser


def load_checkpoint(checkpoint_path, model):
    last_cp= checkpoint_path
    if not os.path.exists(last_cp):
        logger.error(f'checkpoint file {last_cp} not exist, ignore load_checkpoint')
        return
    with open(last_cp,'rb') as f:
        logger.info(f'load checkpoint: {checkpoint_path}')
        state = torch.load(f, map_location=torch.device("cpu"))
    model.load_state_dict(state['model'])
    return model#, state['update_num']


def expand_batch(batch: dict, expand_size: int):
    new_batch = {}
    for k in batch.keys():
        if isinstance(batch[k], list):
            new_batch[k] = batch[k] * expand_size
        else:
            shape_len = len(batch[k].shape)-1
            repeat_shape = [expand_size] + [1] * shape_len
            new_batch[k] = batch[k].repeat(*repeat_shape)

    return new_batch


def main(args):
    config_file = os.path.join(args.root_dir, 'config.json')
    assert os.path.exists(config_file), f'config file not exist: {config_file}'
    config= load_config(config_file)

    # modify config for inference
    config.data.train_mode = False
    config.args = args

    logger.info('start preprocessing...')
    model_config = config.model
    global_config = config.model.global_config

    model = VQStructure(model_config.vqstructure, global_config)

    if args.model_path is not None:
        last_cp = args.model_path
    else:
        checkpoint_dir = f'{args.root_dir}/checkpoint'
        last_cp= os.path.join(checkpoint_dir, f"checkpoint_last.pt")
    logger.info(f'logging model checkpoint')
    _ = load_checkpoint(last_cp, model)

    model.eval()
    model.cuda()

    if os.path.isdir(args.indices_f):
        indices_list = [ f'{args.indices_f}/{f}' for f in os.listdir(args.indices_f)]
    elif os.path.isfile(args.indices_f):
        indices_list = [args.indices_f]

    for indices_f in indices_list:
        if not indices_f.endswith('.npy'):
            continue
        if indices_f.endswith('esm.npy'):
            continue
        indices_dict = np.load(indices_f, allow_pickle=True).item()
        basename = os.path.basename(indices_f).split('.np')[0]
        gen_dir = os.path.dirname(indices_f)

        indices = indices_dict['indices']
        single_res_rel = indices_dict['single_res_rel']
        chain_idx = indices_dict['chain_idx']
        entity_idx = indices_dict['entity_idx']
        pair_res_idx = indices_dict['pair_res_idx']
        pair_chain_idx = indices_dict['pair_chain_idx']
        pair_same_entity = indices_dict['pair_same_entity']

        if len(indices.shape) == 2:
            indices = torch.from_numpy(indices[:, None]).cuda(non_blocking=True)
        elif len(indices.shape) == 1:
            indices = torch.from_numpy(indices[None, None]).cuda(non_blocking=True)
        elif len(indices.shape) == 3:
            indices = torch.from_numpy(indices).cuda(non_blocking=True).permute(1, 0, 2)

        single_mask = torch.ones((1, indices.shape[-1])).cuda(non_blocking=True)
        single_res_rel = torch.from_numpy(single_res_rel)[None].cuda(non_blocking=True)
        chain_idx = torch.from_numpy(chain_idx)[None].cuda(non_blocking=True)
        entity_idx = torch.from_numpy(entity_idx)[None].cuda(non_blocking=True)
        pair_res_idx = torch.from_numpy(pair_res_idx)[None].cuda(non_blocking=True)
        pair_chain_idx = torch.from_numpy(pair_chain_idx)[None].cuda(non_blocking=True)
        pair_same_entity = torch.from_numpy(pair_same_entity)[None].cuda(non_blocking=True)
        try:
            sequence = indices_dict['sequence']
        except:
            sequence = None

        _ = model.gen_structure_from_indices(
            indices, single_mask, 
            single_res_rel, 
            chain_idx, 
            entity_idx, 
            pair_res_idx, 
            pair_chain_idx, 
            pair_same_entity, 
            prefix=f'{gen_dir}/{basename}_last', 
            save_dict=False,
            sequence=sequence)



if __name__=='__main__':
    parser= build_parser()
    args= parser.parse_args()
    main(args)

