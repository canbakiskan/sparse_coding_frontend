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

When a file (dictionary, checkpoint, attack) is saved, all the relevant settings are encoded into the name of the file.

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
6. Default defense (T=15) using DCT basis instead of the overcomplete basis

Run the corresponding bash script in `shell_scripts/ablation` to reproduce.

## Complete List of Settings/Hyperparameters
### General Settings/Hyperparameters
|Setting|Location|Default value|Choices|Help|
|--|--|--|--|--|
|save_checkpoint|`config.toml`|true|bool|Whether to save checkpoint after training|
|seed|`config.toml`|2021|int|Seed for random generator|
|use_gpu|`config.toml`|true|bool|Whether to use Nvidia GPU or run everything on the CPU|

### Dataset Related Settings/Hyperparameters
|Setting|Location|Default value|Choices|Help|
|--|--|--|--|--|
|img_shape|`config.toml`|[32, 32, 3]|[x,x,3]|Height x Width x Channel|
|name|`config.toml`|"CIFAR10"|"CIFAR10", "Imagenet", "Imagenette", "Tiny-ImageNet"|Name of the dataset|
|nb_classes|`config.toml`|10|int|Number of classes in the dataset|

### Dictionary Related Settings/Hyperparameters
|Setting|Location|Default value|Choices|Help|
|--|--|--|--|--|
|batch_size|`config.toml`|5|int|Batch size when learning the dictionary|
|display|`config.toml`|false|bool|Whether to save a plot of first 100 dictionary atoms|
|iter|`config.toml`|1000|int|Number of iterations in dictionary learning optimization|
|lamda|`config.toml`|1.0|float|Coefficient in front of L1 norm of code, see https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.MiniBatchDictionaryLearning.html|
|nb_atoms|`config.toml`|500|int|Number of atoms in the dictionary|
|online|`config.toml`|false|bool|Whether to use online learning. If true, memory footprint is small but it takes a very long time.|
|type|`config.toml`|"overcomplete"|"overcomplete","dct"|Type of dictionary. Overcomplete is the default case, DCT is used for ablation|

### Defense Related Settings/Hyperparameters
|Setting|Location|Default value|Choices|Help|
|--|--|--|--|--|
|activation_beta|`config.toml`|3.0|float|Threshold of the activation function = beta*assumed_budget|
|assumed_budget|`config.toml`|0.03137254901|float|Threshold of the activation function = beta*assumed_budget|
|dropout_p|`parameters.py`|0.95|float|If dropout is used in the encoder, probability of dropping coefficients|
|frontend_arch|`parameters.py`|"top_T_activation_frontend"|"top_T"? + "dropout"? + "activation"? + "noisy"? + {"", "small", "deep", "resize", "identity"} + "frontend" where ? means presence or absence|Contains all the operations that are part of the encoder and the decoder type|
|patch_size|`config.toml`|4|int|Spatial size of the patches usually 4 for CIFAR10, 8 for Imagenet(te)|
|stride|`config.toml`|2|int|Stride while extracting patches|
|test_noise_gamma|`parameters.py`|0.0|float|If the encoder is noisy, what is the coefficient in front of uniform test noise. Noise = gamma * assumed_budget * U[-1,1]|
|top_T|`parameters.py`|15|int|How many top coefficients to keep, if top T taking operation is part of the encoder|
|train_noise_gamma|`parameters.py`|100.0|float|If the encoder is noisy, what is the coefficient in front of uniform train noise. Noise = gamma * assumed_budget * U[-1,1]|

### Neural-net Related Settings/Hyperparameters
|Setting|Location|Default value|Choices|Help|
|--|--|--|--|--|
|classifier_arch|`parameters.py`|"resnet"|"resnet","resnetwide","efficientnet","preact_resnet", "dropout_resnet", "resnet_after_encoder"|"resnet":ResNet-32, "resnetwide":x10 wider version of ResNet-32, "efficientnet": EfficientNet-B0, "preact_resnet": preactivation ResNet-101, "dropout_resnet": ResNet-32 with dropout after the first layer (for ablation), "resnet_after_encoder": ResNet-32 with 500 filters right after encoder (for ablation)|
|epochs|`parameters.py`|70|int|Number of training epochs|
|no_frontend|`parameters.py`|false|bool|Whether to have no frontend (for benchmarks)|
|optimizer.lr|`parameters.py`|0.01|float|Learning rate|
|optimizer.lr_scheduler|`parameters.py`|"cyc"|"cyc","mult","step"|Learning rate scheduler. "cyc":cylic, "mult": multiplicative, "step": x0.1 every 50th epoch|
|optimizer.momentum|`config.toml`|0.9|float|Coefficient of momentum, if optimizer uses it|
|optimizer.name|`config.toml`|"sgd"|"sgd","adam","rms"|"sgd": SGD, "adam": ADAM, "rms": RMSProp|
|optimizer.weight_decay|`config.toml`|0.0005|float|L2 regularizer coefficient|
|optimizer.cyclic.lr_max|`parameters.py`|0.02|float|Maximum learning rate if cyclic learning rate is used|
|optimizer.cyclic.lr_min|`config.toml`|0.0|float|Minumum learning rate if cyclic learning rate is used|
|save_checkpoint|`config.toml`|true|bool|Whether to save the trained model as a checkpoint|
|test_batch_size|`config.toml`|100|int|Inference batch size|
|train_batch_size|`config.toml`|64|int|Training batch size|


### Adversarial Training Related Settings/Hyperparameters
|Setting|Location|Default value|Choices|Help|
|--|--|--|--|--|
|budget|`config.toml`|0.03137254901|float|Attack budget epsilon, in adversarial training|
|EOT_size|`config.toml`|10|int|If EOT is used, number of gradients to average over, in adversarial training|
|method|`parameters.py`|"PGD"|PGD_EOT","PGD_smooth","FGSM","RFGSM","PGD","PGD_EOT","PGD_EOT_normalized","PGD_EOT_sign","CWlinf_EOT",            "CWlinf_EOT_normalized","CWlinf"|Attack method|
|nb_restarts|`config.toml`|1|int|If a variant of PGD or CWlinf is used number of random restarts, in adversarial training|
|nb_steps|`config.toml`|10|int|Number of attack steps, in adversarial training|
|norm|`config.toml`|"inf"|"inf",int|Attack norm (p in L^p), in adversarial training|
|rand|`config.toml`|true|bool|Whether to use random restarts, in adversarial training. Defaults to true if nb_restarts>1|
|rfgsm_alpha|`config.toml`|0.03921568627|float|Alpha in RFGSM|
|step_size|`config.toml`|0.00392156862|float|Attack step size delta, in adversarial training|

### Adversarial Testing (Attack) Related Settings/Hyperparameters
|Setting|Location|Default value|Choices|Help|
|--|--|--|--|--|
|activation_backward|`parameters.py`|"smooth"|"smooth","identity"|If activation is used in the encoder, what should the backward pass approximation be|
|activation_backward_steepness|`parameters.py`|4.0|float|If activation_backward=smooth, how smooth the approximation should be (higher less smooth)|
|box_type|`parameters.py`|"white"|"white","other"|Is the attack whitebox or some type of blackbox|
|budget|`parameters.py`|0.03137254901960784|float|Attack budget epsilon|
|dropout_backward|`parameters.py`|"default"|"identity","default"|If dropout is used in the encoder, what should the backward pass be|
|EOT_size|`parameters.py`|40|int|If EOT is used in the attack, number of gradients to average over|
|method|`parameters.py`|"PGD"|PGD_EOT","PGD_smooth","FGSM","RFGSM","PGD","PGD_EOT","PGD_EOT_normalized","PGD_EOT_sign","CWlinf_EOT",            "CWlinf_EOT_normalized","CWlinf"|Attack method|
|nb_imgs|`config.toml`| -1|uint,-1|Number of test images to process|
|nb_restarts|`parameters.py`|100|int|If a variant of PGD or CWlinf is used number of random restarts|
|nb_steps|`parameters.py`|40|int|Number of attack steps|
|norm|`parameters.py`|"inf"|"inf",int|Attack norm (p in L^p)|
|otherbox_type|`parameters.py`|"boundary"|"transfer", "boundary", "hopskip", "genattack"|Blackbox attack type|
|progress_bar|`config.toml`|true|bool|Whether to show progess bar in the command line while attacking|
|rand|`parameters.py`|true|bool|Whether to use random restarts. Defaults to true if nb_restarts>1|
|rfgsm_alpha|`config.toml`|0.03921568627|float|Alpha in RFGSM|
|save|`config.toml`|true|bool|Whether to save the generated attack|
|skip_clean|`config.toml`|false|bool|Whether to display clean accuracy before attacking|
|step_size|`parameters.py`|0.00392156862745098|float|Attack step size delta|
|top_T_backward|`parameters.py`|"top_U"|"default", "top_U"|If top T operation is used in the encoder, what should the backward pass be. "top_U" means gradients are propagated through top U coefficients|
|top_U|`parameters.py`|30|int|If top_T_backward="top_U", the value of U|
|transfer_file|`parameters.py`|None|filepath|If otherbox_type="transfer", which attack file to apply|

### Ablation Related Settings/Hyperparameters
|Setting|Location|Default value|Choices|Help|
|--|--|--|--|--|
|no_dictionary|`config.toml`|false|bool|Whether to use learnable parameters instead of the dictionary in the encoder (ablation)|

## Decoding File Names

## Folder Structure 

Repository structure is as follows:

```
Repository
│   LICENSE
│   README.md
│   requirements.txt            Required python libraries to run codes
│	
└───checkpoints                 Saved checkpoints
│   │ 
│   └───classifiers  
│   │   └───CIFAR10  
│   │   └───Imagenet  
│   │   └───Imagenette  
│   │   └───Tiny-ImageNet
│   │   
│   └───frontends        
│       └───CIFAR10  
│       └───Imagenet  
│       └───Imagenette  
│       └───Tiny-ImageNet
│	
└───data
│   │ 
│   └───attacked_datasets       Saved attacks
│   │   └───CIFAR10  
│   │   └───Imagenet  
│   │   └───Imagenette  
│   │   └───Tiny-ImageNet
│   │
│   └───dictionaries            Saved dictionaries
│   │   └───CIFAR10  
│   │   └───Imagenet  
│   │   └───Imagenette  
│   │   └───Tiny-ImageNet
│   │
│   └───image_distances         Saved test image L2 distances to one another (for blackbox attack initialization)
│   │   └───CIFAR10  
│   │   └───Imagenet  
│   │   └───Imagenette  
│   │   └───Tiny-ImageNet
│   │
│   └───original_datasets       
│       └───CIFAR10  
│       └───Imagenet  
│       └───Imagenette  
│       └───Tiny-ImageNet
│	
└───figs
│	
└───logs
│	
└───src     
    │   config.toml                          Less frequently changing settings
    │   find_img_distances.py                Find closest images for decision boundary attack
    │   learn_patch_dict.py                  Sparse dictionary learning
    │   parameters.py                        Frequently changing settings
    │   run_attack.py                        Evaluate attacks on models
    │   run_attack_foolbox.py                Similar to run_attack, with foolbox
    │   train_test_functions.py              Train/test helper functions
    │   train.py                             Trains the classifier and the frontend
    │
    │───models
    │   │   bpda.py 	                     Backward pass differentiable approximation implementations
    │   │   combined.py                      Model that combines frontend and classifier
    │   │   decoders.py                      Different decoder definitions
    │   │   efficientnet.py                  EfficientNet definition
    │   │   encoder.py                       Different encoder definitions
    │   │   frontend.py 	                 Different frontend definitions
    │   │   preact_resnet.py                 Pre-activation ResNet definition
    │   │   resnet.py                        ResNet and Wide ResNet definition
    │   │   tools.py                         Tools/functions used in models
    │   │
    │   └───ablation
    │       │   dct_generate                 To generate DCT basis
    │       │   dropout_resnet.py            ResNet with dropout in the first layer
    │       │   resnet_after_encoder.py      ResNet with 500 filters in the first layer
    │       │   sparse_frontend.py           Sparse frontend definition
    │
    └───plotting
    │   │   accuracy_vs_eps.py               Plots accuracy vs attack budget
    │   │   activation_bpda.py               Plots activation backward approximations
    │   │   correlataion_plot.py             Plots first layer outputs of different models
    │   │   dct_hist.py                      will be removed 
    │   │   loss_landscape.py                Plots loss landscapes
    │   │   plot_attacks.py                  Plots generated attacks
    │
    └───shell_scripts
    │   │ 
    │   └───ablation                         See Ablation section above
    │   │   │   1.sh                         
    │   │   │   2.sh
    │   │   │   3.sh
    │   │   │   4.sh
    │   │   │   5.sh
    │   │   │   6.sh
    │   │   │   base_commands.sh             Helper for other scripts
    │   │   
    │   └───by_attack
    │   │   │   boundary.sh                  Runs boundary attack on all models
    │   │   │   defense_args.sh              Helper for other scripts
    │   │   │   L1.sh                        Runs L1 PGD attack on all models
    │   │   │   L2.sh                        Runs L2 PGD attack on all models
    │   │   │   Linf_CW.sh                   Runs Linf Carlini Wagner PGD attack on all models
    │   │   │   Linf.sh                      Runs Linf Carlini Wagner PGD attack on all models
    │   │   │   train.sh                     Trains all models
    │   │   
    │   └───by_defense
    │       │   base_commands.sh             Helper for other scripts
    │       │   natural.sh                   Trains and attacks vanilla ResNet-32
    │       │   our_defense.sh               Trains and attacks model with our defense
    │       │   PGD_AT.sh                    Trains and attacks adversarially trained with PGD
    │       │   TRADES.sh                    Trains and attacks adversarially trained with TRADES
    │
    └───utils
        │   get_modules.py                   Reads and loads checkpoints, dictionaries
        │   get_optimizer_scheduler.py       Returns optimizer and scheduler              
        │   namers.py                        Constructs appropriate name for checkpoints, dictionaries, attacks
        │   plot_settings.py                 Helper for pretty plots
        │   powerset.py                      Generates power set of list
        │   read_datasets.py                 Reads and returns datasets


```

## License

Apache License 2.0
