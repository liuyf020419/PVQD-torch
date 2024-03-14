# PVQD
Protein structure Vector Quantization Diffusion model.

PVQD is a method based on deep learning for protein structure design and prediction. Here we published the source code and the demos for PVQD. This code is developed on Torch framework.

## Install Dependencies
First, please install the depandencies of PVQD
```
conda create -n PVQD python=3.8
conda activate PVQD

pip install -r ./install/requirements.txt
bash ./install/postInstall.sh
```
Then, download the weight from https://biocomp.ustc.edu.cn/servers/downloads/PVQD_ckpt.tar.gz, and extract the archive files `PVQD_ckpt.tar.gz`.

To predict structure from primary amino acid sequence, please additionally install ESM package through following commands and download the weight from https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt.
```
pip install fair-esm  # latest release, OR:
pip install git+https://github.com/facebookresearch/esm.git  # bleeding edge, current repo main branch
```

To generate structure with PVQD+SCUBA-D approach, please additionally install SCUBA-D package through following comands and download the weight from https://biocomp.ustc.edu.cn/servers/downloads/checkpoint_clean.pt.

```
https://github.com/liuyf020419/SCUBA-D.git
```

## Quick start

Three scripts related to the manuscript are saved in `scripts`:
* `sample_from_fa.sh` - structure prediction
* `sample_uncond.sh` - unconditional structure generation
* `structure_to_ary.sh` - encode and decode protein structure to generate an latent space array

The results of demos are saved in `results`.