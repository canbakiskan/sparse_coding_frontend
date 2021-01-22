#!/bin/bash 

COMMAND="python -m neuro-inspired-defense.src.train_classifier --autoencoder_arch=top_T_dropout_quant_resize_autoencoder --dataset=Imagenette --defense_patchsize=8 --defense_stride=4 --dict_nbatoms=1000 --dict_iter=10000 --dict_lambda=0.5 --top_T=100 --dropout_p=0.95 --test_batch_size=32 --autoencoder_train_supervised --classifier_arch=efficientnet --classifier_epoch=100"
echo $COMMAND
eval $COMMAND

COMMAND="python -m neuro-inspired-defense.src.run_attack --autoencoder_arch=top_T_dropout_quant_resize_autoencoder --dataset=Imagenette --defense_patchsize=8 --defense_stride=4 --dict_nbatoms=1000 --dict_iter=10000 --dict_lambda=0.5 --top_T=100 --dropout_p=0.95 --test_batch_size=32 --autoencoder_train_supervised --classifier_arch=efficientnet --classifier_epoch=100"
echo $COMMAND
eval $COMMAND

COMMAND="python -m neuro-inspired-defense.src.run_attack --attack_whitebox_type=W-NFGA --autoencoder_arch=top_T_dropout_quant_resize_autoencoder --dataset=Imagenette --defense_patchsize=8 --defense_stride=4 --dict_nbatoms=1000 --dict_iter=10000 --dict_lambda=0.5 --top_T=100 --dropout_p=0.95 --test_batch_size=32 --autoencoder_train_supervised --classifier_arch=efficientnet --classifier_epoch=100"
echo $COMMAND
eval $COMMAND

COMMAND="python -m neuro-inspired-defense.src.run_attack --attack_whitebox_type=W-NFGA --attack_num_steps=100 --attack_step_size=0.00196 --autoencoder_arch=top_T_dropout_quant_resize_autoencoder --dataset=Imagenette --defense_patchsize=8 --defense_stride=4 --dict_nbatoms=1000 --dict_iter=10000 --dict_lambda=0.5 --top_T=100 --dropout_p=0.95 --test_batch_size=32 --autoencoder_train_supervised --classifier_arch=efficientnet --classifier_epoch=100"
echo $COMMAND
eval $COMMAND

COMMAND="python -m neuro-inspired-defense.src.run_attack --autoencoder_arch=top_T_dropout_quant_resize_autoencoder  --dataset=Imagenette --defense_patchsize=8 --defense_stride=4 --dict_nbatoms=1000 --dict_iter=10000 --dict_lambda=0.5 --top_T=100 --dropout_p=0.95 --test_batch_size=32 --classifier_arch=efficientnet --classifier_epochs=100 --attack_box_type=other --attack_transfer_file=white_W-AIGA_PGD_EOT_normalized_eps_4_Ne_40_Ns_20_ss_0_Nr_1_efficientnet_sgd_cyc_0.0500_ep_100_ps_8_st_4_l_0.5_n_1000_it_10000_top_T_dropout_quant_resize_autoencoder_sgd_cyc_0.0500_T_100_j_24_p_0.95_US_ep_50.npy --autoencoder_train_supervised"
echo $COMMAND
eval $COMMAND
