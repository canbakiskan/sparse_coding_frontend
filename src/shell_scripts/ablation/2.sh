#!/bin/bash

full_path=$(realpath $0)
dir_path=$(dirname $full_path)

source "$dir_path/base_commands.sh"
defense_args="--autoencoder_arch=top_T_autoencoder"

for base_command in "${base_commands[@]}"
do

eval "$base_command $defense_args"

done