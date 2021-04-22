#!/bin/bash

full_path=$(realpath $0)
dir_path=$(dirname $full_path)

source "$dir_path/base_commands.sh"
defense_args="--defense_frontend_arch=top_T_frontend --defense_top_T=500 --attack_top_U=500 --neural_net_lr_max=0.01"

for base_command in "${base_commands[@]}"
do

eval "$base_command $defense_args"

done
