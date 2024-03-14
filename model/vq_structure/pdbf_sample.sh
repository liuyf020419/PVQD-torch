#!/bin/bash
name=rvq_4_512_16_trans4_3ipa6_fdinit_1e4_mu11_init_rvq_newclamp_senc_640

testlist=/home/liuyf/proteins/mc_vq_20230919/experiment/complex_vq/utils/data_list/pc95_res_0-2_1024.txt
# testlist=/home/liuyf/proteins/mc_vq_20230919/experiment/monomer_vq/utils/data_list/pc40.0_res0.0-2.0_noBrks_len40-10000_Xray_d2022_10_28_chains12908_00
# testlist=/home/liuyf/proteins/mc_vq_20230919/complex_vq/test_complex/test.txt

pdbroot=/database/assemble/divided
# pdbroot=/home/liuyf/database/datanew
# pdbroot=/home/liuyf/proteins/mc_vq_20230919/complex_vq/test_complex/pdb

gen_dir=/home/liuyf/proteins/mc_vq_20230919/experiment/complex_vq/reconstruct_nat_pdb_complex_new_1024
# gen_dir=/home/liuyf/proteins/mc_vq_20230919/experiment/complex_vq/reconstruct_nat_pdb_complex_new
# gen_dir=/home/liuyf/proteins/mc_vq_20230919/experiment/complex_vq/reconstruct_nat_pdb_new
# gen_dir=/home/liuyf/proteins/mc_vq_20230919/complex_vq/test_complex/gen

CUDA_VISIBLE_DEVICES=1 python3.8 inference_pdbf_data.py \
    --root_dir savedir/${name} \
    --test_list ${testlist} \
    --gen_dir ${gen_dir} \
    --max_sample_num 10000000 \
    --pdb_root ${pdbroot} \
    --write_pdbfile \
    --batch_size 1 \
    --num_workers 20 \
    --return_structure \
    --model_path savedir/${name}/checkpoint/checkpoint_last_new.pt