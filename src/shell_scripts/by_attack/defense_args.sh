#!/bin/bash

our_defense=""
natural="--neural_net_epochs=100 --neural_net_no_frontend --neural_net_lr_max=0.05"
PGD_AT="--neural_net_epochs=100 --neural_net_no_frontend --adv_training_method=PGD --neural_net_epochs=100 --neural_net_lr_max=0.05"
TRADES="--neural_net_epochs=100 --neural_net_no_frontend --adv_training_method=TRADES --neural_net_epochs=76 --neural_net_lr0.01 --lr_scheduler=step"

defense_args=("$our_defense" "$natural" "$PGD_AT" "$TRADES")
