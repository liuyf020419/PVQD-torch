#!/bin/bash
name=structure_vq

testlist="../data/reconstruct_pdb.txt" # pdbcode, chain (C1+C2), length (default 1)
pdbroot="../data/pdb_f"
gen_dir="../results/reconstruct"

CUDA_VISIBLE_DEVICES=0 python3.8 ../model/vq_structure/inference_pdbf_data.py \
    --root_dir "../ckpt/${name}" \
    --test_list ${testlist} \
    --gen_dir ${gen_dir} \
    --max_sample_num 10000000 \
    --pdb_root ${pdbroot} \
    --write_pdbfile \
    --batch_size 1 \
    --num_workers 20 \
    --return_structure \
    --model_path "../ckpt/${name}/checkpoint/checkpoint_last.pt"