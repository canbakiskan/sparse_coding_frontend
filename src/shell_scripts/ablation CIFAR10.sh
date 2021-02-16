#!/bin/bash 

# default noiseless
COMMAND="python -m NID_main.src.train_classifier --autoencoder_train_supervised --autoencoder_arch=top_T_quant_autoencoder --top_T=1"
echo $COMMAND
eval $COMMAND

COMMAND="python -m NID_main.src.run_attack --autoencoder_train_supervised --autoencoder_arch=top_T_quant_autoencoder \
--top_T=1 --attack_box_type=other --attack_otherbox_type=transfer \
--attack_transfer_file=white_W-AIGA_PGD_EOT_normalized_eps_8_Ne_40_Ns_20_ss_1_Nr_1_resnet_sgd_cyc_0.0500_ep_70_ps_4_st_2_l_1.0_n_500_it_1000_top_T_dropout_autoencoder_sgd_cyc_0.0500_T_50_p_0.95_US_ep_50.npy"
echo $COMMAND
eval $COMMAND


# no dictionary used; encoder weights are trainable calisti ama %55 falan accuracy
COMMAND="python -m NID_main.src.train_classifier --autoencoder_train_supervised --autoencoder_arch=top_T_quant_autoencoder --top_T=1 --ablation_no_dictionary"
echo $COMMAND
eval $COMMAND

# top 2
COMMAND="python -m NID_main.src.train_classifier --autoencoder_train_supervised --autoencoder_arch=top_T_quant_autoencoder --top_T=2"
echo $COMMAND
eval $COMMAND

# top 5 learning rate i dusurdum
COMMAND="python -m NID_main.src.train_classifier --autoencoder_train_supervised --autoencoder_arch=top_T_quant_autoencoder --top_T=5 --lr_max=0.02"
echo $COMMAND
eval $COMMAND

# top 10 learning rate i dusurdum
COMMAND="python -m NID_main.src.train_classifier --autoencoder_train_supervised --autoencoder_arch=top_T_quant_autoencoder --top_T=10 --lr_max=0.02"
echo $COMMAND
eval $COMMAND

# no top learning rate i dusurdum
COMMAND="python -m NID_main.src.train_classifier --autoencoder_train_supervised --autoencoder_arch=quant_autoencoder --lr_max=0.02"
echo $COMMAND
eval $COMMAND

# cnn decoder is removed, encoder outputs go directly into classifier
COMMAND="python -m NID_main.src.train_classifier --autoencoder_train_supervised --autoencoder_arch=top_T_quant_identity_autoencoder --top_T=1 --classifier_arch=resnet_after_encoder"
echo $COMMAND
eval $COMMAND

# no activation function
COMMAND="python -m NID_main.src.train_classifier --autoencoder_train_supervised --autoencoder_arch=top_T_autoencoder --top_T=1"
echo $COMMAND
eval $COMMAND

# no activation function or top T taking
COMMAND="python -m NID_main.src.train_classifier --autoencoder_train_supervised --autoencoder_arch=top_T_autoencoder --top_T=500"
echo $COMMAND
eval $COMMAND
