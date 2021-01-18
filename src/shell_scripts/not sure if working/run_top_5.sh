#!/bin/bash 

COMMAND="python train_autoencoder.py --autoencoder_arch=top_T_quant_autoencoder --top_T=5"
echo $COMMAND
eval $COMMAND

COMMAND="python train_classifier.py --autoencoder_arch=top_T_quant_autoencoder --top_T=5"
echo $COMMAND
eval $COMMAND

COMMAND="python run_attack.py --autoencoder_arch=top_T_quant_autoencoder --top_T=5 --attack_method=PGD"
echo $COMMAND
eval $COMMAND

COMMAND="python run_attack.py --autoencoder_arch=top_T_quant_autoencoder --top_T=5 --attack_method=PGD  --attack_num_steps=100 --attack_step_size=0.00196 "
echo $COMMAND
eval $COMMAND

COMMAND="python run_attack.py --autoencoder_arch=top_T_quant_autoencoder --top_T=5 --attack_method=PGD --attack_num_restarts=10"
echo $COMMAND
eval $COMMAND

COMMAND="python run_attack.py --autoencoder_arch=top_T_quant_autoencoder --top_T=5 --attack_method=CWlinf"
echo $COMMAND
eval $COMMAND


COMMAND="python run_attack.py --autoencoder_arch=top_T_quant_autoencoder --top_T=5 --attack_method=PGD --attack_whitebox_type=W-NFGA"
echo $COMMAND
eval $COMMAND

COMMAND="python run_attack.py --autoencoder_arch=top_T_quant_autoencoder --top_T=5 --attack_method=PGD --attack_whitebox_type=SW"
echo $COMMAND
eval $COMMAND
