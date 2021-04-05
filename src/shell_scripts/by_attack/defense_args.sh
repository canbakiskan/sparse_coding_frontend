#!/bin/bash

our_defense=""
natural="--neural_net_epochs=100 --neural_net_no_frontend"
PGD_AT="--neural_net_epochs=100 --neural_net_no_frontend --adv_training_method=PGD"
TRADES="--neural_net_epochs=100 --neural_net_no_frontend --adv_training_method=TRADES"

defense_args=("$our_defense" "$natural" "$PGD_AT" "$TRADES")
