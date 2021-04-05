#!/bin/bash

full_path=$(realpath $0)
dir_path=$(dirname $full_path)

shellscripts_path="$(dirname $dir_path)"
src_path="$(dirname $shellscripts_path)"

if grep -E "lr_max\s*=\s*0.01" $src_path/config.toml; then
    source "$dir_path/base_commands.sh"
    defense_args="--defense_frontend_arch=top_T_activation_identity_frontend"

    for base_command in "${base_commands[@]}"
    do

    eval "$base_command $defense_args"

    done
else
    echo "Before running this script, please change neural_net.classifier_arch to \"resnet_after_encoder\" in config.toml."
fi