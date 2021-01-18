#!/bin/bash 

export CUDA_VISIBLE_DEVICES=1
#python run_attack.py --autoencoder_arch=top_T_quant_autoencoder  --autoencoder_train_supervised --attack_whitebox_type=W-NFGA --attack_quantization_BPDA_steepness=4 --attack_skip_clean --lr_max=0.01 --attack_method=PGD
#python run_attack.py --autoencoder_arch=top_T_quant_autoencoder  --autoencoder_train_supervised --attack_skip_clean --lr_max=0.01 --attack_method=PGD


python run_attack.py --dropout_p=0.75 --top_T=10 --autoencoder_train_supervised --attack_whitebox_type=W-NFGA --attack_skip_clean  --attack_quantization_BPDA_steepness=4 --lr_max=0.025
python run_attack.py --dropout_p=0.75 --top_T=10 --autoencoder_train_supervised --lr_max=0.025      