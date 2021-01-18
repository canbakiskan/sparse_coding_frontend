#!/bin/bash 

COMMAND="python train_classifier.py --no_autoencoder --classifier_epochs=100 "
echo $COMMAND
eval $COMMAND

COMMAND="python run_attack.py --no_autoencoder --classifier_epochs=100 --attack_method=PGD"
echo $COMMAND
eval $COMMAND

COMMAND="python run_attack.py --no_autoencoder --classifier_epochs=100 --attack_method=PGD --attack_num_steps=100 --attack_step_size=0.00196 "
echo $COMMAND
eval $COMMAND

COMMAND="python run_attack.py --no_autoencoder --classifier_epochs=100 --attack_method=PGD --attack_num_restarts=10"
echo $COMMAND
eval $COMMAND

COMMAND="python run_attack.py --no_autoencoder --classifier_epochs=100 --attack_method=CWlinf"
echo $COMMAND
eval $COMMAND

COMMAND="python run_attack.py --no_autoencoder --classifier_epochs=100 --attack_method=CWlinf --attack_num_restarts=100 --attack_num_steps=100 --attack_step_size=0.00196"
echo $COMMAND
eval $COMMAND
