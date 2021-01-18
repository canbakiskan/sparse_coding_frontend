#!/bin/bash 

COMMAND="python train_classifier.py --autoencoder_train_supervised"
echo $COMMAND
eval $COMMAND

COMMAND="python run_attack.py --autoencoder_train_supervised --attack_method=PGD"
echo $COMMAND
eval $COMMAND

COMMAND="python run_attack.py --autoencoder_train_supervised"
echo $COMMAND
eval $COMMAND


COMMAND="python run_attack.py --autoencoder_train_supervised --attack_EOT_size=100"
echo $COMMAND
eval $COMMAND

COMMAND="python run_attack.py --autoencoder_train_supervised --attack_num_steps=100 --attack_step_size=0.00196 "
echo $COMMAND
eval $COMMAND

COMMAND="python run_attack.py --autoencoder_train_supervised --attack_method=CWlinf_EOT_normalized"
echo $COMMAND
eval $COMMAND

COMMAND="python run_attack.py --autoencoder_train_supervised --attack_whitebox_type=W-NFGA"
echo $COMMAND
eval $COMMAND

COMMAND="python run_attack.py --autoencoder_train_supervised --attack_whitebox_type=SW"
echo $COMMAND
eval $COMMAND

"rm /home/adv/AAAI/data/attacked_dataset/CIFAR10/PW-T.npy"
"ln -s /home/adv/AAAI/data/attacked_dataset/CIFAR10/white_W-AIGA_PGD_eps_8_Ns_20_ss_1_Nr_1_resnet_sgd_cyc_0.0500_ep_70_ps_4_st_2_l_1.0_n_500_it_1000_top_T_quant_autoencoder_sgd_cyc_0.0500_T_50_j_24_US_ep_50.npy /home/adv/AAAI/data/attacked_dataset/CIFAR10/PW-T.npy"
"python run_attack.py --autoencoder_train_supervised --attack_box_type=other --attack_otherbox_type=PW-T --autoencoder_arch=top_T_quant_autoencoder --lr_max=0.01"


"rm /home/adv/AAAI/data/attacked_dataset/CIFAR10/PW-T.npy"
"ln -s /home/adv/AAAI/data/attacked_dataset/CIFAR10/white_W-AIGA_PGD_EOT_normalized_eps_8_Ne_40_Ns_20_ss_1_Nr_1_resnet_sgd_cyc_0.0500_ep_70_ps_4_st_2_l_1.0_n_500_it_1000_top_T_dropout_quant_autoencoder_sgd_cyc_0.0500_T_500_j_24_p_0.95_US_ep_50.npy /home/adv/AAAI/data/attacked_dataset/CIFAR10/PW-T.npy"
"python run_attack.py --autoencoder_train_supervised --attack_box_type=other --attack_otherbox_type=PW-T --autoencoder_arch=top_T_dropout_quant_autoencoder --top_T=500 "


"ln -s /home/adv/AAAI/data/attacked_dataset/Imagenette/white_W-AIGA_PGD_eps_4_Ns_20_ss_1_Nr_1_efficientnet_sgd_cyc_0.0500_ep_100_ps_8_st_4_l_0.5_n_1000_it_10000_top_T_dropout_quant_resize_autoencoder_sgd_cyc_0.0500_T_100_j_24_p_0.95_US_ep_50.npy /home/adv/AAAI/data/attacked_dataset/Imagenette/PW-T.npy"
"python run_attack.py --autoencoder_arch=top_T_dropout_quant_resize_autoencoder  --dataset=Imagenette --defense_patchsize=8 --defense_stride=4 --dict_nbatoms=1000 --dict_iter=10000 --dict_lambda=0.5 --top_T=100 --dropout_p=0.95 --test_batch_size=32 --classifier_arch=efficientnet --classifier_epochs=70 --autoencoder_train_supervised --attack_box_type=other --attack_otherbox_type=PW-T"

"rm /home/adv/AAAI/data/attacked_dataset/Imagenette/PW-T.npy "
"ln -s /home/adv/AAAI/data/attacked_dataset/Imagenette/white_W-AIGA_PGD_eps_8_Ns_20_ss_1_Nr_1_efficientnet_sgd_cyc_0.0500_ep_100_ps_8_st_4_l_0.5_n_1000_it_10000_top_T_dropout_quant_resize_autoencoder_sgd_cyc_0.0500_T_100_j_24_p_0.95_US_ep_50.npy /home/adv/AAAI/data/attacked_dataset/Imagenette/PW-T.npy"
"python run_attack.py --autoencoder_arch=top_T_dropout_quant_resize_autoencoder  --dataset=Imagenette --defense_patchsize=8 --defense_stride=4 --dict_nbatoms=1000 --dict_iter=10000 --dict_lambda=0.5 --top_T=100 --dropout_p=0.95 --test_batch_size=32 --classifier_arch=efficientnet --classifier_epochs=70 --autoencoder_train_supervised --attack_box_type=other --attack_otherbox_type=PW-T"