#!/bin/bash
name=crystal_nmr_model

fasta_f="../data/demo.fasta" # fasta file for prediction

CUDA_VISIBLE_DEVICES=0 python3.8 ../model/folding_diff/inference_from_fa.py \
    --root_dir "../ckpt/structure_pred/${name}" \
    --fasta_f ${fasta_f} \
    --step_size 1 \
    --batch_size 1 \
    --model_path "../ckpt/structure_pred/${name}/checkpoint/checkpoint_last.pt" \
    --diff_noising_scale 0.1 \
    --gen_dir "../results/prediction" \
    --esm_script "../utils/esm_extract.py" \
    --esm_param "/database/lyf_database/pretrain_lm/esm/param/esm2_t33_650M_UR50D.pt" \
    --decoder_root "../model/vq_structure" \
    --decoder_param "../ckpt/structure_vq/checkpoint/checkpoint_last.pt" \
    --esm_rep_extract 
