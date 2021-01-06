#!/bin/bash 

COMMAND="python -m neuro-inspired-defense.src.train_autoencoder.py"
echo $COMMAND
eval $COMMAND

COMMAND="python -m neuro-inspired-defense.src.train_classifier.py"
echo $COMMAND
eval $COMMAND

COMMAND="python -m neuro-inspired-defense.src.run_attack.py"
echo $COMMAND
eval $COMMAND

COMMAND="python -m neuro-inspired-defense.src.run_attack.py --attack_whitebox_type=W-NFGA"
echo $COMMAND
eval $COMMAND

COMMAND="python -m neuro-inspired-defense.src.run_attack.py --attack_whitebox_type=W-NFGA --attack_num_steps=100 --attack_step_size=0.00196 "
echo $COMMAND
eval $COMMAND
