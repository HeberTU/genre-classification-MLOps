Genre Classification with MLOps
==============================
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

- [Origin](https://github.com/HeberTU/genre-classification-MLOps)
- Author: Heber Trujillo <heber.trj.urt@gmail.com>
- Date of last README.md update: 01.04.2022

## Repo Overview

This repository shows how to implement an end-to-end machine learning pipeline using [mlflow](https://mlflow.org/) 
and [W&B](https://wandb.ai/site).

## How to Run Scripts 

### Dependencies Installation 

1. Create and activate a virtual environment for the project. For example:
    ```bash
    python3 -m venv ./.venv
    source ./.venv/bin/activate
    ```
   
2. Install Poetry, the tool used for dependency management. To install it, run from a terminal:
    ```bash
    pip install poetry
    ```

3. From the virtual environment, install the required dependencies with:
    ```bash
    poetry install --no-root
    ```
   
4. Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) for mlflow component 
isolation.