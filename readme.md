# VAE-FMRI-BOLD5000

Implementation of VAE for the fMRI dataset BOLD5000

## Install

1. Clone the repository

```bash
git clone https://github.com/Michedev/deep-learning-template.git
```

2. Install [anaconda](https://www.anaconda.com/) if you don't have it

3. In anaconda shell, download dependencies and data using `anaconda-project run download-data`

## Train

Train the VAE to encode/decode FMRI ROIs data:

```bash
anaconda-project run train-gpu
```

You can also specify additional arguments according to `config/train.yaml` like

```bash
anaconda-project run train-gpu batch_size=64
```


## Project structure

    ├── data  # Data storage folder
    ├── config
    │   ├── dataset  # Dataset config
    │   ├── model  # Model config
    │   ├── model_dataset  # (model, dataset) specific config
    │   ├── test.yaml   # test configuration
    │   └── train.yaml  # train configuration
    ├── dataset  # Dataset definition
    ├── model  # Model definition
    ├── utils
    │   ├── experiment_tools.py # Iterate over experiments
    │   └── paths.py  # common paths
    ├── train.py  # Entrypoint point for training
    ├── test.py  # Entrypoint point for testing
    ├── anaconda-project.yml  # Project configuration
    ├── saved_models  # where models are saved
    └── readme.md  # This file

### Design keypoints
- Root folder should contain only entrypoint
- Add tasks to anaconda-project.yml via the command `anaconda-project add-command`


### Anaconda-project FAQ

#### How to add a new command?
Example:
```bash
anaconda-project add-command generate "python ddpm_pytorch/generate.py
```
#### Mac OS support in lock file

_[Short]_ Run these commands:

```bash
anaconda-project remove-packages cudatoolkit;
anaconda-project add-platforms osx-64;
```

_[Long]_
1. Remove cudatoolkit dependency from _anaconda-project.yml_
```bash
anaconda-project remove-packages cudatoolkit
```
2. Add Mac OS platform to _anaconda-project-lock.yml_:
```bash
anaconda-project add-platforms osx-64
``` 
