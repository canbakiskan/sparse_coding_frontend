#!/bin/bash

full_path=$(realpath $0)
dir_path=$(dirname $full_path)

source "$dir_path/base_commands.sh"
defense_args="--neural_net_no_frontend --adv_training_method=PGD --neural_net_lr_max=0.05"

for base_command in "${base_commands[@]}"
do

eval "$base_command $defense_args"

done