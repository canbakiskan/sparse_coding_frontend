#!/bin/bash 

export CUDA_VISIBLE_DEVICES=0
#python run_attack.py --autoencoder_arch=top_T_dropout_autoencoder  --autoencoder_train_supervised --attack_whitebox_type=W-NFGA --attack_skip_clean
#python run_attack.py --autoencoder_arch=top_T_dropout_autoencoder  --autoencoder_train_supervised --attack_skip_clean

python run_attack.py --dropout_p=0.5 --top_T=5 --autoencoder_train_supervised --attack_whitebox_type=W-NFGA --attack_skip_clean  --attack_quantization_BPDA_steepness=4
python run_attack.py --dropout_p=0.5 --top_T=5 --autoencoder_train_supervised            