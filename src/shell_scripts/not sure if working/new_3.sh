#!/bin/bash 

export CUDA_VISIBLE_DEVICES=3
#python run_attack.py --top_T=500 --autoencoder_train_supervised --attack_whitebox_type=W-NFGA --attack_quantization_BPDA_steepness=4 --attack_skip_clean --lr_max=0.01
#python run_attack.py --top_T=500 --autoencoder_train_supervised --attack_skip_clean --lr_max=0.01

python run_attack.py --autoencoder_arch=sparse_autoencoder --attack_method=PGD --attack_whitebox_type=W-NFGA --autoencoder_train_supervised
python run_attack.py --autoencoder_arch=sparse_autoencoder --autoencoder_train_supervised --attack_whitebox_type=W-NFGA --attack_skip_clean  --attack_quantization_BPDA_steepness=4 --lr_max=0.025
python run_attack.py --dropout_p=0.75 --top_T=10 --autoencoder_train_supervised --lr_max=0.025      