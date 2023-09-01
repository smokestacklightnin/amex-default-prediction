# amex-default-prediction
Kaggle competition where competitors predict if a customer will default in the future

## Installation Instructions

You mus set up the virtual environment, and do the `pip` installation to fully install the project.

### Set up the virtual environment

#### With GPU

In the project directory, run

```
conda env create -f environment_gpu.yml
conda activate amex-gpu
```

#### Without GPU

In the project directory, run

```
conda env create -f environment.yml
conda activate amex
```

### Development Installation

With a conda environment installed and activated, run the following in the project directory to install the project:

```
pip install -e .
```
