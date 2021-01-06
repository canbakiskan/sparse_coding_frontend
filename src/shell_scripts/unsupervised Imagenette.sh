#!/bin/bash 

COMMAND="python train_autoencoder.py --autoencoder_arch=top_T_dropout_quant_resize_autoencoder --dataset=Imagenette --defense_patchsize=8 --defense_stride=4 --dict_nbatoms=1000 --dict_iter=10000 --dict_lambda=0.5 --top_T=100 --dropout_p=0.95 --test_batch_size=32  --classifier_arch=efficientnet --classifier_epoch=100"
echo $COMMAND
eval $COMMAND

COMMAND="python train_classifier.py --autoencoder_arch=top_T_dropout_quant_resize_autoencoder --dataset=Imagenette --defense_patchsize=8 --defense_stride=4 --dict_nbatoms=1000 --dict_iter=10000 --dict_lambda=0.5 --top_T=100 --dropout_p=0.95 --test_batch_size=32  --classifier_arch=efficientnet --classifier_epoch=100"
echo $COMMAND
eval $COMMAND

COMMAND="python run_attack.py --autoencoder_arch=top_T_dropout_quant_resize_autoencoder --dataset=Imagenette --defense_patchsize=8 --defense_stride=4 --dict_nbatoms=1000 --dict_iter=10000 --dict_lambda=0.5 --top_T=100 --dropout_p=0.95 --test_batch_size=32  --classifier_arch=efficientnet --classifier_epoch=100"
echo $COMMAND
eval $COMMAND

COMMAND="python run_attack.py --attack_whitebox_type=W-NFGA --autoencoder_arch=top_T_dropout_quant_resize_autoencoder --dataset=Imagenette --defense_patchsize=8 --defense_stride=4 --dict_nbatoms=1000 --dict_iter=10000 --dict_lambda=0.5 --top_T=100 --dropout_p=0.95 --test_batch_size=32  --classifier_arch=efficientnet --classifier_epoch=100"
echo $COMMAND
eval $COMMAND

COMMAND="python run_attack.py --attack_whitebox_type=W-NFGA --attack_num_steps=100 --attack_step_size=0.00196 --autoencoder_arch=top_T_dropout_quant_resize_autoencoder --dataset=Imagenette --defense_patchsize=8 --defense_stride=4 --dict_nbatoms=1000 --dict_iter=10000 --dict_lambda=0.5 --top_T=100 --dropout_p=0.95 --test_batch_size=32  --classifier_arch=efficientnet --classifier_epoch=100"
echo $COMMAND
eval $COMMAND
