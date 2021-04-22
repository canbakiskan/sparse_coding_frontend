#!/bin/bash

train="python -m sparse_coding_frontend.src.train"
Linf="python -m sparse_coding_frontend.src.run_attack"
Linf_CW="python -m sparse_coding_frontend.src.run_attack --attack_method=CWlinf"
L2="python -m sparse_coding_frontend.src.run_attack --attack_budget=0.6 --attack_step_size=0.075 --attack_norm=2"
L1="python -m sparse_coding_frontend.src.run_attack --attack_budget=30 --attack_step_size=3.75 --attack_norm=1"
boundary="python -m sparse_coding_frontend.src.run_attack --attack_method=boundary"

base_commands=("$train" "$Linf" "$Linf_CW" "$L2" "$L1" "$boundary")
