#!/bin/bash

our_defense=""
natural="--classifier_epochs=100 --no_autoencoder"
PGD_AT="--classifier_epochs=100 --no_autoencoder --adv_training_attack=PGD"
TRADES="--classifier_epochs=100 --no_autoencoder --adv_training_attack=TRADES"

defense_args=("$our_defense" "$natural" "$PGD_AT" "$TRADES")
