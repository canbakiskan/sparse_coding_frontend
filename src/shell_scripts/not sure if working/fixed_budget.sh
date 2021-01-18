#!/bin/bash

budget=50
steps=(3 5 10 20 40)
restarts=(1 2 5 10 20 40)

for attack_num_steps in ${steps[@]}
do
    attack_step_size=$(echo 8/255*2/$attack_num_steps | bc -l)
    
    for attack_num_restarts in ${restarts[@]}
    do  
    
        attack_EOT_size=$(((budget+(attack_num_steps*attack_num_restarts)/2)/(attack_num_steps*attack_num_restarts)))
        
        if [ $attack_EOT_size -ne 0 ];
        then
            echo "steps: $attack_num_steps, stepsize: $attack_step_size, restarts: $attack_num_restarts, EOT: $attack_EOT_size"
            string="python run_attack.py --attack_skip_clean --attack_step_size=$attack_step_size --attack_num_steps=$attack_num_steps --attack_EOT_size=$attack_EOT_size --attack_num_restarts=$attack_num_restarts | grep -E -o \"Saved to .*\""
            path=$(eval $string)
            # echo $path
            python run_attack.py --autoencoder_train_supervised --attack_skip_clean --attack_box_type=other --attack_transfer_file=${path:53}  | grep -E "Attack accuracy: [0-9]+\.[0-9]+\%"
        fi
    done
done