#!/bin/bash

full_path=$(realpath $0)
dir_path=$(dirname $full_path)

source "$dir_path/defense_args.sh"
base_command="python -m sparse_coding_frontend.src.run_attack --attack_method=CWlinf"

for defense_arg in "${defense_args[@]}"
do

eval "$base_command $defense_arg"

done