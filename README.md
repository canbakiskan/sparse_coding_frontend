WARNING: top_T taking was wrongly implemented, fixed in commit 3b4d60aa. This makes already existing checkpoints useless.

# A Neuro-Inspired Autoencoding Defense Against Adversarial Perturbations

This repository is the official implementation of "A Neuro-Inspired Autoencoding Defense Against Adversarial Perturbations" paper at this [link](https://arxiv.org/pdf/2011.10867.pdf). 

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
CIFAR-10 will be downloaded automatically when the code is run.

Following commands assume the name of the folder is `neuro-inspired-defense`.

## Dictionary

To learn the overcomplete dictionary:

```bash
python -m neuro-inspired-defense.src.learn_patch_dict.py
```


## Training

To train the models in the paper, run these commands:

Our defense:
```bash
python -m neuro-inspired-defense.src.train_classifier.py --autoencoder_train_supervised 
```

Our defense, Unsupervised:
```bash
python -m neuro-inspired-defense.src.train_autoencoder.py
python -m neuro-inspired-defense.src.train_classifier.py 
```

## Evaluation

There are many parameters you can use for defense evaluation. For a list of all parameters see `parameters.py`. For default evaluation use:

Our defense:
```bash
python -m neuro-inspired-defense.src.run_attack.py --autoencoder_train_supervised
```
Our defense, Unsupervised:
```bash
python -m neuro-inspired-defense.src.run_attack.py 
```


## Shell Scripts

Alternatively, you can run the bash script for the corresponding model located in the `shell_scripts` directory. These will train the autoencoder and the classifier, and then evaluate attacks using different of parameters.

## Pre-trained Models

You can find pretrained models inside the `checkpoints` directory.


## Folder Structure 

Adversarial framework folder contains the codes for adversarial attacks, analysis, and adversarial training functions. Src folder contains all the necessary codes for autoencoder, training, testing, models, and utility functions. Repository structure is as follows:

```
Repository
│   README.md
│   requirements.txt            Required python libraries to run codes
│	
└───src     
    │   learn_patch_dict.py                  Sparse dictionary learning
    │   parameters.py                        Main file for parameters
    │   run_attack.py                        Evaluate attacks on models
    │   train_autoencoder.py                 Trains the autoencoder
    │   train_classifier.py                  Trains the classifier with or without the autoencoder
    │   train_test_functions.py              Train/test helper functions
    │
    │───models
    │   │   autoencoders.py 	             Different autoencoder definitions
    │   │   bpda.py 	                     Backward pass differentiable approximation model
    │   │   combined.py                      Model that combines autoencoder and clasifier
    │   │   decoders.py                      Different decoder definitions
    │   │   efficientnet.py                  EfficientNet definition
    │   │   encoders.py                      Different encoder definitions
    │   │   ensemble.py                      Ensemble processing model
    │   │   preact_resnet.py                 Pre-activation ResNet definition
    │   │   resnet.py                        ResNet and Wide ResNet definition
    │   │   tools.py                         Tools/functions used in models
    │   └───ablation
    │       │   dropout_resnet.py            ResNet with dropout in first layer
    │       │   find_blur_sigma.py           Find sigma of gaussian filter in gaussian_blur.py
    │       │   gaussian_blur.py             Gaussian blurring preprocessing
    │       │   sparse_autoencoder.py        Sparse autoencoder definition
    │
    └───utils
        │   get_modules.py                   
        │   namers.py
        │   plot_settings.py
        │   read_datasets.py

```

## License

Apache License 2.0
