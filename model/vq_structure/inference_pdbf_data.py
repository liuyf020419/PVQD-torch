import contextlib
import logging
import sys
import json
import ml_collections
from collections import OrderedDict
import time
import sys,os
import argparse
import numpy as np
from tqdm import tqdm, trange

from typing import Any, Dict, List
from ml_collections import ConfigDict
import torch
import torch.nn as nn


from protdiff.models.vqstructure import VQStructure
from protdiff.dataset import VQStructureDatasetnew, DataIterator, GroupedIterator
from protdiff.models.nn_utils import make_mask


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
    parser.add_argument('--test_list', type=str, default=None, help='test list')
    parser.add_argument('--gen_tag', type=str, default='', help='gen_tag')
    parser.add_argument('--seed', type=int, default=1234, help='Seed for PyTorch random number generators')
    parser.add_argument('--sample_from_raw_pdbfile', action='store_true', help='sample from raw pdbfile or processed file')
    parser.add_argument('--max_sample_num', type=int, default=10000000, help='maximum number of samples for testing or application')
    
    parser.add_argument('--write_pdbfile', action='store_true', help='write pdb file when testing model')
    parser.add_argument('--pdb_root', type=str, default=None, help='pdb root for application')
    parser.add_argument('--batch_size', type=int, default=1, help='batchsize for each protein')
    parser.add_argument('--num_workers', type=int, default=1, help='num_workers for dataloader')
    parser.add_argument('--save_all', action='store_true', help='write pdb file when testing model')
    parser.add_argument('--return_structure', action='store_true', help='write pdb file when testing model')

    return parser


def load_config(path)->ml_collections.ConfigDict:
    return ml_collections.ConfigDict(json.loads(open(path).read()))


def apply_to_sample(f, sample):
    if hasattr(sample, "__len__") and len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return tuple(_apply(x) for x in x)
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample, device=None):
    device = device or torch.cuda.current_device()

    def _move_to_cuda(tensor):
        # non_blocking is ignored if tensor is not pinned, so we can always set
        # to True (see github.com/PyTorchLightning/pytorch-lightning/issues/620)
        return tensor.to(device=device, non_blocking=True)

    return apply_to_sample(_move_to_cuda, sample)


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


def load_dataset(config, data_list, batch_size, epoch =0, mode_sel="ALL", num_workers=1, seed=37, shuffle=False,):
    dataset = VQStructureDatasetnew(config, data_list, train=False, pdbroot=args.pdb_root, af2_data=False)
    logger.info(f'loading data {data_list}')
    data_iter=DataIterator(
        dataset,
        num_shards= 1, shard_id= 0,
        epoch= epoch,
        batch_size= batch_size,
        shuffle= shuffle,
        num_workers= num_workers,
        seed = seed
    )
    return data_iter


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
    _, step = load_checkpoint(last_cp, model)

    model.eval()
    model.cuda()

    output_dir= args.gen_dir
    os.makedirs(output_dir, exist_ok=True)

    data_iter = load_dataset(config, args.test_list, args.batch_size, epoch=0, num_workers=args.num_workers)

    data_iter1= data_iter.next_epoch_itr()
    data_iter1= GroupedIterator(data_iter1, chunk_size=1)

    for batch_idx in trange(len(data_iter1)):
        batch = next(data_iter1)
        batch = batch[0]
        if data_iter1.n > args.max_sample_num:
            break

        with torch.no_grad():
            batch = move_to_cuda(batch)
            stacked_codebook_indices = model.sampling(batch, output_dir, return_all=args.return_structure, \
                verbose_indices=False, compute_sc_identity=False, save_rep=args.save_all)
            dtype = batch['gt_pos'].dtype
            batchsize, L, N, _ = batch['gt_pos'].shape
            make_mask(batch['len'], batchsize, L, batch, dtype)

            for batch_idx in range(len(batch['pdbname'])):
                pdbname = batch['pdbname'][batch_idx]
                single_mask = batch['single_mask'][batch_idx]
                
                b_aatype = batch['aatype'][batch_idx][single_mask.bool()].detach().cpu().numpy()
                codebook_indices = stacked_codebook_indices[batch_idx].permute(1, 0)[single_mask.bool()].permute(1, 0) # B, C, N
                b_structrue_ary = np.array(codebook_indices.detach().cpu().numpy()[0].tolist())

                all_data_f = f'{output_dir}/{pdbname}/{pdbname}.npy'
                saved_data_dict = {
                    'pdbname': pdbname,
                    'res_idx': batch['single_res_rel'][batch_idx][single_mask.bool()].detach().cpu().numpy(),
                    'aatype': b_aatype,
                    'structure_ary': b_structrue_ary
                }
                np.save(all_data_f, saved_data_dict)


if __name__=='__main__':
    parser= build_parser()
    args= parser.parse_args()
    main(args)

