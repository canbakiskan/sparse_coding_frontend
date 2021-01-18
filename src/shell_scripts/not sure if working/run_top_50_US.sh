#!/bin/bash 

COMMAND="python train_autoencoder.py"
echo $COMMAND
eval $COMMAND

COMMAND="python train_classifier.py"
echo $COMMAND
eval $COMMAND

COMMAND="python run_attack.py --attack_method=PGD"
echo $COMMAND
eval $COMMAND

COMMAND="python run_attack.py"
echo $COMMAND
eval $COMMAND

COMMAND="python run_attack.py --attack_EOT_size=100"
echo $COMMAND
eval $COMMAND

COMMAND="python run_attack.py --attack_num_steps=100 --attack_step_size=0.00196 "
echo $COMMAND
eval $COMMAND

COMMAND="python run_attack.py --attack_num_restarts=10"
echo $COMMAND
eval $COMMAND

COMMAND="python run_attack.py --attack_method=CWlinf_EOT_normalized"
echo $COMMAND
eval $COMMAND

COMMAND="python run_attack.py --attack_whitebox_type=W-NFGA"
echo $COMMAND
eval $COMMAND

COMMAND="python run_attack.py --attack_whitebox_type=SW"
echo $COMMAND
eval $COMMAND

