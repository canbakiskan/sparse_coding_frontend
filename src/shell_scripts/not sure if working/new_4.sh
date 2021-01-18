#!/bin/bash

pid=320605 
while [ -d /proc/$pid ] ; do
    sleep 1
done 

export CUDA_VISIBLE_DEVICES=0
python run_attack.py --attack_skip_clean --attack_whitebox_type=W-NFGA --attack_quantization_BPDA_steepness=4