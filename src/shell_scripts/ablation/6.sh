#!/bin/bash

full_path=$(realpath $0)
dir_path=$(dirname $full_path)

shellscripts_path="$(dirname $dir_path)"
src_path="$(dirname $shellscripts_path)"

if grep -E "type\s*=\s*[',\"]dct[',\"]" $src_path/config.toml; then
    source "$dir_path/base_commands.sh"
    defense_args=""

    for base_command in "${base_commands[@]}"
    do

    eval "$base_command $defense_args"

    done
else
    echo "Before running this script, please change dictionary.type to \"dct\" in config.toml"
fi