#!/bin/bash

full_path=$(realpath $0)
dir_path=$(dirname $full_path)

source "$dir_path/defense_args.sh"
base_command="python -m sparse_coding_frontend.src.run_attack --attack_budget=0.6 --attack_step_size=0.075 --attack_norm=2"

for defense_arg in "${defense_args[@]}"
do

eval "$base_command $defense_arg"

done