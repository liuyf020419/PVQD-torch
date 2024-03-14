import contextlib
import imp
import logging
import sys
from collections import OrderedDict, defaultdict
import time
import sys,os,json
import subprocess
import argparse
from tqdm import tqdm
import numpy as np

from typing import Any, Dict, List
from ml_collections import ConfigDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from protdiff.models.ddpm import DDPM

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
    parser.add_argument('--root_dir', type=str, default=None, help='project path')
    parser.add_argument('--gen_tag', type=str, default='', help='gen_tag')
    parser.add_argument('--seed', type=int, default=1234, help='Seed for PyTorch random number generators')
    parser.add_argument('--fasta_f', type=str, required=True, help='maximum number of samples for testing or application')
    
    parser.add_argument('--step_size', type=int, default=3, help='iteration number per epoch')
    parser.add_argument('--batch_size', type=int, default=1, help='batchsize for each protein')
    parser.add_argument('--diff_noising_scale', type=float, default=0.1, help='noising scale for diffusion')
    parser.add_argument('--mapping_nn', action='store_true', help='sample from raw pdbfile or processed file')
    parser.add_argument('--esm_script', type=str, required=True, help='esm_script')
    parser.add_argument('--esm_param', type=str, required=True, help='esm_param')
    parser.add_argument('--esm_rep_extract', action='store_true', help='esm_rep_extract')
    parser.add_argument('--decoder_root', type=str, required=True, help='decoder_root')
    parser.add_argument('--decoder_param', type=str, required=True, help='decoder_param')

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
    return model


def extract_esm2_rep(esm_script, esm_param, fasta_f, gen_dir):
    str_decoder_script = os.path.abspath(esm_script)
    esm2_ckpt = esm_param
    
    command = [
        "python3.8", str_decoder_script, 
        f'{esm2_ckpt}', f'{os.path.abspath(fasta_f)}',
        f'{os.path.abspath(gen_dir)}'
        ]
    subprocess.run(command)


def add_assembly_feature(chain_lens, merged_pdbresID, seq_str):
    rel_all_chain_features = {}
    seq_to_entity_id = {}
    grouped_chains_length = defaultdict(list)
    chain_length_summed = 0
    for chain_len in chain_lens:
        start_index = chain_length_summed
        chain_length_summed += int(chain_len)
        end_index = chain_length_summed
        seq = seq_str[start_index: end_index]
        if seq not in seq_to_entity_id:
            seq_to_entity_id[seq] = len(seq_to_entity_id) + 1
        grouped_chains_length[seq_to_entity_id[seq]].append(chain_len)

    asym_id_list, sym_id_list, entity_id_list, num_sym_list = [], [], [], []
    chain_id = 0
    for entity_id, group_chain_features in grouped_chains_length.items():
        num_sym = len(group_chain_features)  # zy
        for sym_id, seq_length in enumerate(group_chain_features, start=1):
            asym_id_list.append(chain_id * torch.ones(seq_length)) # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
            sym_id_list.append(sym_id * torch.ones(seq_length)) # [1,2,3,1,2,3,1,2,3,1,2,3,1,2,3]
            entity_id_list.append(entity_id * torch.ones(seq_length)) # [1,1,1,2,2,2,3,3,3,4,4,4,5,5,5]
            num_sym_list.append(num_sym * torch.ones(seq_length)) # [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
            chain_id += 1

    rel_all_chain_features['asym_id'] = torch.cat(asym_id_list)
    rel_all_chain_features['sym_id'] = torch.cat(sym_id_list)
    rel_all_chain_features['entity_id'] = torch.cat(entity_id_list)
    rel_all_chain_features['num_sym'] = torch.cat(num_sym_list)
    rel_all_chain_features['res_id'] = merged_pdbresID

    return rel_all_chain_features


def make_multichains_rel_pos(chain_rel_pos_dict: str, rmax=32, smax=5):
    # pair pos
    diff_aym_id = (chain_rel_pos_dict['asym_id'][None, :] - chain_rel_pos_dict['asym_id'][:, None])
    diff_res_id = (chain_rel_pos_dict['res_id'][None, :] - chain_rel_pos_dict['res_id'][:, None])
    diff_sym_id = (chain_rel_pos_dict['sym_id'][None, :] - chain_rel_pos_dict['sym_id'][:, None])
    diff_entity_id = (chain_rel_pos_dict['entity_id'][None, :] - chain_rel_pos_dict['entity_id'][:, None])

    clamp_res_id = torch.clamp(diff_res_id+rmax, min=0, max=2*rmax)
    pair_res_idx = torch.where(diff_aym_id.long() == 0, clamp_res_id.long(), 2*rmax+1) # 2*rmax + 2

    same_chain = (chain_rel_pos_dict['asym_id'][None, :] == chain_rel_pos_dict['asym_id'][:, None]).long()
    same_entity = (chain_rel_pos_dict['entity_id'][None, :] == chain_rel_pos_dict['entity_id'][:, None]).long() # 2 + 1

    clamp_sym_id = torch.clamp(diff_sym_id+smax, min=0, max=2*smax)
    pair_chain_idx = torch.where(diff_entity_id.long() == 0, clamp_sym_id.long(), 2*smax+1) # 2*smax + 2

    pair_rel_pos_dict = {
        'pair_res_idx': pair_res_idx,
        'pair_same_entity': same_entity,
        'pair_chain_idx': pair_chain_idx,
        'pair_same_chain': same_chain,
        'single_res_rel': chain_rel_pos_dict['res_id'],
        'chain_idx': chain_rel_pos_dict['asym_id'],
        'entity_idx': chain_rel_pos_dict['entity_id']-1
    }
    return pair_rel_pos_dict


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


def parse_fasta(fasta_f):
    with open(fasta_f, 'r') as reader:
        all_lines = reader.readlines()
    fasta_dict = {}
    for l_idx, line in enumerate(all_lines):
        if line.startswith('>'):
            query = line.strip()[1:]
            seq = all_lines[l_idx+1].strip()
            fasta_dict[query] = seq

    return fasta_dict


def retrive_structure(decoder_root, decoder_param, gen_dir, ):
    decoder_root = os.path.abspath(decoder_root)
    decoder_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(decoder_param)))
    str_decoder_script = f'{decoder_root}/inference_from_indices.py'
    command = [
        "python3.8", str_decoder_script, 
        "--root_dir", decoder_root_dir,
        "--max_sample_num", '10000000',
        "--indices_f", gen_dir,
        "--write_pdbfile",
        "--batch_size", '1',
        "--model_path", decoder_param
        ]
    subprocess.run(command)



def main(args):
    config_file = os.path.join(args.root_dir, 'config.json')
    assert os.path.exists(config_file), f'config file not exist: {config_file}'
    config= load_config(config_file)

    # modify config for inference
    config.data.train_mode = False
    config.args = args

    logger.info('start preprocessing...')
    model_config = config.model.latent_diff_model
    global_config = config.model.global_config

    model = DDPM(model_config, global_config)

    if args.model_path is not None:
        last_cp = args.model_path
    else:
        checkpoint_dir = f'{args.root_dir}/checkpoint'
        last_cp= os.path.join(checkpoint_dir, f"checkpoint_last.pt")
    logger.info(f'logging model checkpoint')
    _ = load_checkpoint(last_cp, model)

    model.eval()
    model.cuda()
    model.x0_pred_net.eval()

    sub_dir = os.path.basename(args.fasta_f).split('.fa')[0]
    output_dir= args.gen_dir

    output_dir = f'{output_dir}/{sub_dir}_' + str(args.diff_noising_scale)
    if args.mapping_nn:
        output_dir = output_dir + '_mapnn'
        
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    fasta_dict = parse_fasta(args.fasta_f)
    if args.esm_rep_extract:
        extract_esm2_rep(args.esm_script, args.esm_param, args.fasta_f, output_dir)
        logger.info(f'esm rep extracted')
    
    for (query, seq) in tqdm(fasta_dict.items()):
        protein_length = len(seq)
        data = {}
        data['pdbname'] = f'{query}'
        data['len'] = torch.tensor(protein_length)
        data['aatype'] = torch.zeros(protein_length)
        data['gt_backbone_pos'] = torch.rand(protein_length, 5, 3)
        data['gt_backbone_frame'] = torch.rand(protein_length, 8, 12)
        data['gt_backbone_pos_atom37'] = torch.rand(protein_length, 37, 3)
        data['sidechain_function_pos'] = torch.rand(protein_length, 3, 3)
        data['sidechain_function_coords_mask'] = torch.ones(protein_length, 3).bool()
        data['sstype'] = torch.zeros(protein_length).long()
        data['protein_state'] = torch.zeros(1).long()
        data['atom37_mask'] = torch.ones(protein_length, 37)

        data['condition_mask'] = torch.ones(protein_length).long()
        data_dict = np.load(f'{output_dir}/{query}/{query}_esm.npy', allow_pickle=True).item()
        try:
            esm2_rep = data_dict['esm2']['representations'][33]
        except:
            esm2_rep = data_dict['esm2']['representations'][str(33)]
        data['esm_rep'] = esm2_rep

        merged_pdbresID = torch.arange(protein_length)
        fake_sequence_str = 'A'*protein_length
        chain_rel_pos_dict = add_assembly_feature(
            [protein_length], merged_pdbresID, fake_sequence_str)
        pair_rel_pos_dict = make_multichains_rel_pos(chain_rel_pos_dict)
        data.update(pair_rel_pos_dict)

        pdbname = data['pdbname']
        batch = {}

        for k, v in data.items():
            if k not in ['len', 'loss_mask', 'pdbname', 'noising_mode_idx', 'cath_architecture', 'reduced_chain_idx', 'chain_mask_str']:
                batch[k] = v[None].cuda(non_blocking=True)
            elif k in ['pdbname', 'noising_mode_idx', 'reduced_chain_idx', 'chain_mask_str']:
                batch[k] = [v]
            else:
                try:
                    batch[k] = v.cuda(non_blocking=True)
                except:
                    print(k)
                    import pdb; pdb.set_trace()
        
        pdb_prefix = f'{output_dir}/{pdbname}/{pdbname}'
        logger.info(f'pdb name: {pdbname}; length: {protein_length}')
        os.makedirs(f'{output_dir}/{pdbname}', exist_ok=True)

        logger.info(f'generating {pdbname} ...')

        with torch.no_grad():
            if (not os.path.isfile(f'{output_dir}/{pdbname}/{pdbname}_{args.batch_size-1}.npy')):
                batch = expand_batch(batch, args.batch_size)  
                print(torch.where(batch['condition_mask'][0]))              
                x0_dict = model.sampling(batch, pdb_prefix, args.step_size, mapping_nn=args.mapping_nn)
                l2_distance = x0_dict['l2_distance']
                batchsize, num_res, str_code_num = l2_distance.shape
                gen_token = torch.argmin(l2_distance, dim=-1)

                pdb_feature_dict = {
                    'single_res_rel': batch['single_res_rel'][0].detach().cpu().numpy(),
                    'chain_idx': batch['chain_idx'][0].detach().cpu().numpy(),
                    'entity_idx': batch['entity_idx'][0].detach().cpu().numpy(),
                    'pair_res_idx': batch['pair_res_idx'][0].detach().cpu().numpy(),
                    'pair_chain_idx': batch['pair_chain_idx'][0].detach().cpu().numpy(),
                    'pair_same_entity': batch['pair_same_entity'][0].detach().cpu().numpy()
                }
                os.makedirs(f'{output_dir}/{pdbname}', exist_ok=True)
                for b_idx in range(batchsize):
                    pdb_feature_dict['indices'] = gen_token[b_idx].detach().cpu().numpy()
                    pdb_feature_dict['sequence'] = seq
                    np.save(f'{output_dir}/{pdbname}/{pdbname}_{b_idx}.npy', pdb_feature_dict)
                    
            if (not os.path.isfile(f'{output_dir}/{pdbname}/{pdbname}_{args.batch_size-1}_last_vqgen_from_indice.pdb')):
                retrive_structure(args.decoder_root, args.decoder_param, f'{output_dir}/{pdbname}')

if __name__=='__main__':
    parser= build_parser()
    args= parser.parse_args()
    main(args)

