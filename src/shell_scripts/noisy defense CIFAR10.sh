#!/bin/bash 

COMMAND="python -m NID_main.src.train_classifier --autoencoder_arch=top_T_quant_noisy_autoencoder --top_T=1 --noise_gamma=5.0 --autoencoder_train_supervised"
echo $COMMAND
eval $COMMAND

COMMAND="python -m NID_main.src.run_attack --autoencoder_arch=top_T_quant_noisy_autoencoder --top_T=1 --noise_gamma=5.0 --autoencoder_train_supervised"
echo $COMMAND
eval $COMMAND

COMMAND="python -m NID_main.src.run_attack --autoencoder_arch=top_T_quant_noisy_autoencoder --top_T=1 --noise_gamma=5.0 --autoencoder_train_supervised --attack_whitebox_type=W-NFGA"
echo $COMMAND
eval $COMMAND

COMMAND="python -m NID_main.src.run_attack --autoencoder_arch=top_T_quant_noisy_autoencoder --top_T=1 --noise_gamma=5.0 --autoencoder_train_supervised --attack_whitebox_type=W-NFGA --attack_num_steps=100 --attack_step_size=0.00196 "
echo $COMMAND
eval $COMMAND

COMMAND="python -m NID_main.src.run_attack --autoencoder_arch=top_T_quant_noisy_autoencoder --top_T=1 --noise_gamma=100.0 --autoencoder_train_supervised --attack_box_type=other --attack_otherbox_type=transfer --attack_transfer_file=white_W-AIGA_PGD_EOT_normalized_eps_8_Ne_40_Ns_20_ss_1_Nr_1_resnet_sgd_cyc_0.0500_ep_70_ps_4_st_2_l_1.0_n_500_it_1000_top_T_dropout_autoencoder_sgd_cyc_0.0500_T_50_p_0.95_US_ep_50.npy"
echo $COMMAND
eval $COMMAND
