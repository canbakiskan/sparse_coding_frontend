#!/bin/bash

full_path=$(realpath $0)
dir_path=$(dirname $full_path)

shellscripts_path="$(dirname $dir_path)"
src_path="$(dirname $shellscripts_path)"

if grep -E "lr_max\s*=\s*0.01" $src_path/config.toml; then
    source "$dir_path/base_commands.sh"
    defense_args="--autoencoder_arch=top_T_autoencoder --top_T=500 --top_U=500"

    for base_command in "${base_commands[@]}"
    do

    eval "$base_command $defense_args"

    done
else
    echo "Before running this script, please change neural_net.optimizer.cyclic.lr_max to 0.01 in config.toml. Otherwise it doesn't converge"
fi