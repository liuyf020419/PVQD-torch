#!/bin/bash
name=uncond_gen

monomer_len=100 # protein_length

CUDA_VISIBLE_DEVICES=1 python3.8 "../model/uncond_diff/inference_uncond.py" \
    --root_dir "../ckpt/${name}" \
    --monomer_length ${monomer_len} \
    --step_size 1 \
    --batch_size 1 \
    --model_path "../ckpt/${name}/checkpoint/checkpoint_last.pt" \
    --gen_dir "../results/unconditional_gen" \
    --diff_noising_scale 0.1 \
    --decoder_root "../model/vq_structure" \
    --decoder_param "../ckpt/structure_vq/checkpoint/checkpoint_last.pt" \
    --SCUBAD_root "../../SCUBA-D-git/code" # SCUBA-D working dir