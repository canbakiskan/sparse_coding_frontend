# Sparse Coding Frontend for Robust Neural Networks

This repository is the official implementation of "Sparse Coding Frontend for Robust Neural Networks" paper published in ICLR 2021 Workshop on Security and Safety in Machine Learning Systems. If you have questions you can contact canbakiskan@ucsb.edu.

## Requirements

Install the requirements:

```bash
pip install -r requirements.txt
```

Then add current directory to `$PYTHONPATH`:

```bash
export PYTHONPATH="${PYTHONPATH}:/path/containing/project/directory"
```

Download Imagenette [here](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz), and place in the `data/original_dataset/`. Then extract it using `tar -xvzf imagenette2-160.tgz`. CIFAR10 will be downloaded automatically when the code is run.

Following commands assume the name of the folder is `sparse_coding_frontend`.

## Configuration Files

`config.toml` and `parameters.py` contain the hyperparameters and other configuration settings related to the project. Settings that change less frequently are stored in `config.toml`. Settings that change more frequently are stored in `parameters.py` and can be specified as command line arguments when any python file is run.

When a file (dictionary, checkpoint, attack) is saved all the relevant settings are encoded into the name of the file.

The default dataset is CIFAR10, if you would like to change the dataset, comment out the relevant lines in `config.toml`.


## Dictionary Learning

To learn the overcomplete dictionary, run:

```bash
python -m sparse_coding_frontend.src.learn_patch_dict
```

The relevant parameters for dictionary learning are all listed in `config.toml` under the `dictionary` table. Learnt dictionaries for CIFAR10, Imagenet and Imagenette are already present in the repository under `data/dictionaries/{dataset}`.

## Shell Scripts

To reproduce the results in Table 1 of the paper, you can run the bash scripts located in the `shell_scripts` directory. Use scripts in `by_defense` if you want to reproduce rows, use scripts in `by_attack` if you want ro reproduce columns (after running `train.sh`). 

## Pre-trained Models

You can find pretrained models inside the `checkpoints` directory.

## Ablation

For ablation the following cases are considered and implemented:

1. Default defense with T=1,2,5,10,20,50
2. Default defense (T=15) without activation
3. Default defense without top_T operation (essentially T=500) and without activation (otherwise it doesn't converge)
4. Default defense (T=15) without the decoder (encoder outputs fed into ResNet widened at the first layer)
5. Default defense (T=15) without the overcomplete dictionary (learnable weights instead)
6. Default defense (T=15) using DCT basis instead of the sparsifying overcomplete basis

Run the corresponding bash script in `shell_scripts/ablation` to reproduce.

## Complete List of Settings/Hyperparameters
### General Settings/Hyperparameters
|Setting|Where|Default|Choices|Help|
|--|--|--|--|--|
|log_interval|`config.toml`|100|int||
|save_checkpoint|`config.toml`|true|bool||
|seed|`config.toml`|2021|int||
|use_gpu|`config.toml`|true|bool||

### Dataset Related Settings/Hyperparameters
|Setting|Where|Default|Choices|Help|
|--|--|--|--|--|
|img_shape|`config.toml`|[32, 32, 3]|[x,x,3]||
|name|`config.toml`|false|bool||
|nb_classes|`config.toml`|10|int||

### Dictionary Related Settings/Hyperparameters
|Setting|Where|Default|Choices|Help|
|--|--|--|--|--|
|batch_size|`config.toml`|5|int||
|display|`config.toml`|false|bool||
|iter|`config.toml`|1000|int||
|lamda|`config.toml`|1.0|float||
|nb_atoms|`config.toml`|500|int||
|online|`config.toml`|false|bool||
|type|`config.toml`|"overcomplete"|"overcomplete","dct"||

### Defense Related Settings/Hyperparameters
|Setting|Where|Default|Choices|Help|
|--|--|--|--|--|
|activation_beta|`config.toml`|3.0|float||
|assumed_budget|`config.toml`|0.03137254901|
|dropout_p|`parameters.py`|0.95|float||
|frontend_arch|`parameters.py`|"top_T_activation_frontend"|
|patch_size|`config.toml`|4|int||
|stride|`config.toml`|2|int||
|test_noise_gamma|`parameters.py`|0.0|float||
|top_T|`parameters.py`|15|int||
|train_noise_gamma|`parameters.py`|100.0|float||

### Neural-net Related Settings/Hyperparameters
|Setting|Where|Default|Choices|Help|
|--|--|--|--|--|
|classifier_arch|`parameters.py`|"resnet"|
|epochs|`parameters.py`|70|int||
|no_frontend|`parameters.py`|false|bool||
|optimizer.lr|`config.toml`|0.001|float||
|optimizer.lr_scheduler|`config.toml`|"cyc"|"cyc","mult","step"||
|optimizer.momentum|`config.toml`|0.9|float||
|optimizer.name|`config.toml`|"sgd"|"sgd","adam","rms"||
|optimizer.weight_decay|`config.toml`|0.0005|float||
|optimizer.cyclic.lr_max|`config.toml`|0.02|float||
|optimizer.cyclic.lr_min|`config.toml`|0.0|float||
|save_checkpoint|`config.toml`|true|bool||
|test_batch_size|`config.toml`|100|int||
|train_batch_size|`config.toml`|64|int||


### Adversarial Training Related Settings/Hyperparameters
|Setting|Where|Default|Choices|Help|
|--|--|--|--|--|
|budget|`config.toml`|0.03137254901|float||
|EOT_size|`config.toml`|10|int||
|method|`parameters.py`|"PGD"|PGD_EOT","PGD_smooth","FGSM","RFGSM","PGD","PGD_EOT","PGD_EOT_normalized","PGD_EOT_sign","CWlinf_EOT",            "CWlinf_EOT_normalized","CWlinf"||
|nb_restarts|`config.toml`|1|int||
|nb_steps|`config.toml`|10|int||
|norm|`config.toml`|"inf"|"inf",int||
|rand|`config.toml`|true|bool||
|rfgsm_alpha|`config.toml`|0.03921568627|float||
|step_size|`config.toml`|0.00392156862|float||

### Adversarial Testing (Attack) Related Settings/Hyperparameters
|Setting|Where|Default|Choices|Help|
|--|--|--|--|--|
|activation_backward|`parameters.py`|"smooth"|"smooth","identity"||
|activation_backward_steepness|`parameters.py`|4.0|float||
|box_type|`parameters.py`|"white"|"white","other"||
|budget|`parameters.py`|0.03137254901960784|float||
|dropout_backward|`parameters.py`|"default"|"identity","default"||
|EOT_size|`parameters.py`|40|int||
|method|`parameters.py`|"PGD"|PGD_EOT","PGD_smooth","FGSM","RFGSM","PGD","PGD_EOT","PGD_EOT_normalized","PGD_EOT_sign","CWlinf_EOT",            "CWlinf_EOT_normalized","CWlinf"||
|nb_imgs|`config.toml`| -1|uint,-1||
|nb_restarts|`parameters.py`|100|int||
|nb_steps|`parameters.py`|40|int||
|norm|`parameters.py`|"inf"|"inf",int||
|otherbox_type|`parameters.py`|"boundary"|"transfer", "boundary", "hopskip", "genattack"||
|progress_bar|`config.toml`|true|bool||
|rand|`parameters.py`|true|bool||
|rfgsm_alpha|`config.toml`|0.03921568627|float||
|save|`config.toml`|true|bool||
|skip_clean|`config.toml`|false|bool||
|step_size|`parameters.py`|0.00392156862745098|float||
|top_T_backward|`parameters.py`|"top_U"|"default", "top_U"||
|top_U|`parameters.py`|30|int||
|transfer_file|`parameters.py`|None|filepath||

### Ablation Related Settings/Hyperparameters
|Setting|Where|Default|Choices|Help|
|--|--|--|--|--|
|distill|`config.toml`|false|bool||
|no_dictionary|`config.toml`|false|bool||

## Decoding File Names

## Folder Structure 

Adversarial framework folder contains the codes for adversarial attacks, analysis, and adversarial training functions. Src folder contains all the necessary codes for frontend, training, testing, models, and utility functions. Repository structure is as follows:

```
Repository
│   LICENSE
│   README.md
│   requirements.txt            Required python libraries to run codes
│	
└───checkpoints
│	
└───data
│	
└───figs
│	
└───logs
│	
└───src     
    │   config.toml                          Less frequently changing settings
    │   dct_generate                         To generate DCT basis (ablation)
    │   find_img_distances.py                Find closest images for decision boundary attack
    │   learn_patch_dict.py                  Sparse dictionary learning
    │   parameters.py                        Frequently changing settings
    │   run_attack.py                        Evaluate attacks on models
    │   run_attack_foolbox.py                Similar to run_attack, with foolbox
    │   train_test_functions.py              Train/test helper functions
    │   train.py                             Trains the classifier and the frontend
    │
    │───models
    │   │   bpda.py 	                     Backward pass differentiable approximation model
    │   │   combined.py                      Model that combines frontend and clasifier
    │   │   decoders.py                      Different decoder definitions
    │   │   efficientnet.py                  EfficientNet definition
    │   │   encoder.py                      Different encoder definitions
    │   │   frontend.py 	                 Different frontend definitions
    │   │   preact_resnet.py                 Pre-activation ResNet definition
    │   │   resnet.py                        ResNet and Wide ResNet definition
    │   │   tools.py                         Tools/functions used in models
    │   └───ablation
    │       │   distill_attack.py            ResNet with dropout in first layer
    │       │   distill_train.py           Sparse frontend definition
    │       │   dropout_resnet.py           Sparse frontend definition
    │       │   resnet_after_encoder.py           Sparse frontend definition
    │       │   sparse_frontend.py           Sparse frontend definition
    │
    └───plotting
    │   │   accuracy_vs_eps.py     
    │   │   activation_bpda.py
    │   │   before_after_frontend.py
    │   │   correlataion_plot.py
    │   │   dct_hist.py
    │   │   loss_landscape.py
    │   │   plot_attacks.py
    │
    └───shell_scripts
    │   │   our_defense.sh     
    │   │   natural.sh
    │   │   PGD_AT.sh
    │   │   TRADES.sh
    │
    └───utils
        │   get_modules.py     
        │   get_optimizer_scheduler.py                   
        │   namers.py
        │   plot_settings.py
        │   powerset.py
        │   read_datasets.py


```

## License

Apache License 2.0
