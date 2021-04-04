# Sparse Coding Frontend for Robust Neural Networks

This repository is the official implementation of "Sparse Coding Frontend for Robust Neural Networks" paper published in ICLR 2021 Workshop on Security and Safety in Machine Learning Systems

## Requirements

Install the requirements:

```bash
pip install -r requirements.txt
```

Then add current directory to `$PYTHONPATH`:

```bash
export PYTHONPATH="${PYTHONPATH}:/path/containing/project/directory"
```

Download Imagenette [here](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz), and place in the `data/original_dataset/`. Then extract it using `tar -xvzf imagenette2-160.tgz`.
CIFAR10 will be downloaded automatically when the code is run.

Following commands assume the name of the folder is `sparse_coding_frontend`.

## Dictionary

To learn the overcomplete dictionary:

```bash
python -m sparse_coding_frontend.src.learn_patch_dict.py
```


## Training

To train the models in the paper, run these commands:

Our defense:
```bash
python -m sparse_coding_frontend.src.train.py  
```

## Evaluation

There are many parameters you can use for defense evaluation. For a list of all parameters see `parameters.py`. For default evaluation use:

Our defense:
```bash
python -m sparse_coding_frontend.src.run_attack.py 
```
Our defense, Unsupervised:
```bash
python -m sparse_coding_frontend.src.run_attack.py 
```


## Shell Scripts

Alternatively, you can run the bash script for the corresponding model located in the `shell_scripts` directory. These will train the frontend and the classifier, and then evaluate attacks using different of parameters.

## Pre-trained Models

You can find pretrained models inside the `checkpoints` directory.


## Folder Structure 

Adversarial framework folder contains the codes for adversarial attacks, analysis, and adversarial training functions. Src folder contains all the necessary codes for frontend, training, testing, models, and utility functions. Repository structure is as follows:

```
Repository
│   README.md
│   requirements.txt            Required python libraries to run codes
│	
└───src     
    │   learn_patch_dict.py                  Sparse dictionary learning
    │   parameters.py                        Main file for parameters
    │   run_attack.py                        Evaluate attacks on models
    │   train_frontend.py                 Trains the frontend
    │   train_classifier.py                  Trains the classifier with or without the frontend
    │   train_test_functions.py              Train/test helper functions
    │
    │───models
    │   │   frontends.py 	             Different frontend definitions
    │   │   bpda.py 	                     Backward pass differentiable approximation model
    │   │   combined.py                      Model that combines frontend and clasifier
    │   │   decoders.py                      Different decoder definitions
    │   │   efficientnet.py                  EfficientNet definition
    │   │   encoder.py                      Different encoder definitions
    │   │   ensemble.py                      Ensemble processing model
    │   │   preact_resnet.py                 Pre-activation ResNet definition
    │   │   resnet.py                        ResNet and Wide ResNet definition
    │   │   tools.py                         Tools/functions used in models
    │   └───ablation
    │       │   dropout_resnet.py            ResNet with dropout in first layer
    │       │   sparse_frontend.py        Sparse frontend definition
    │
    └───utils
        │   get_modules.py                   
        │   namers.py
        │   plot_settings.py
        │   read_datasets.py

```

## License

Apache License 2.0
