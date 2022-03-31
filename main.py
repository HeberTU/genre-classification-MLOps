# -*- coding: utf-8 -*-
"""Orchestrator scrips.

Created on: 3/31/2022
@author: Heber Trujillo <heber.trj.urt@gmail.com> 
Licence,
"""
import mlflow
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.listconfig import ListConfig


@hydra.main(config_name='config')
def main(config: DictConfig):
    """Pipeline Orchestrator."""

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    root_path = hydra.utils.get_original_cwd()

    if isinstance(config["main"]["execute_steps"], str):
        steps_to_execute = config["main"]["execute_steps"].split(",")
    else:
        assert isinstance(config["main"]["execute_steps"], ListConfig)
        steps_to_execute = config["main"]["execute_steps"]

    if "download" in steps_to_execute:
        _ = mlflow.run(
            uri=os.path.join(root_path, "download"),
            entry_point="main",
            parameters={
                "file_url": config["data"]["file_url"],
                "artifact_name": "raw_data.parquet",
                "artifact_type": "raw_data",
                "artifact_description": "Data as downloaded"
            }
        )

    if "preprocess" in steps_to_execute:
        _ = mlflow.run(
            uri=os.path.join(root_path, "preprocess"),
            entry_point="main",
            parameters={
                "input_artifact": "raw_data.parquet:latest",
                "artifact_name": "preprocessed_data.csv",
                "artifact_type": "clean_data",
                "artifact_description": "Data after preprocessing"
            }
        )

    if "check_data" in steps_to_execute:
        _ = mlflow.run(
            uri=os.path.join(root_path, "check_data"),
            entry_point="main",
            parameters={
                "reference_artifact": config["data"]["reference_dataset"],
                "sample_artifact": 'preprocessed_data.csv:latest',
                "ks_alpha": config["data"]["ks_alpha"]
            }
        )

    if "segregate" in steps_to_execute:

        _ = mlflow.run(
            uri=os.path.join(root_path, "segregate"),
            entry_point="main",
            parameters={
                "input_artifact": 'preprocessed_data.csv:latest',
                "artifact_root": 'data',
                "artifact_type": 'segregated_data',
                "test_size": config["data"]["test_size"],
                "random_state": config["main"]["random_seed"],
                "stratify": config["data"]["stratify"],
            }
        )

    if "random_forest" in steps_to_execute:
        # Serialize decision tree configuration
        model_config = os.path.abspath("random_forest_config.yml")

        with open(model_config, "w+") as fp:
            fp.write(OmegaConf.to_yaml(config["random_forest_pipeline"]))

        _ = mlflow.run(
            uri=os.path.join(root_path, "random_forest"),
            entry_point="main",
            parameters={
                "train_data": 'data_train.csv:latest',
                "model_config": model_config,
                "export_artifact": 'model',
                "random_seed": config["main"]["random_seed"],
                "val_size": config["data"]["val_size"],
                "stratify": config["data"]["stratify"]
            }
        )

    if "evaluate" in steps_to_execute:
        _ = mlflow.run(
            uri=os.path.join(root_path, "evaluate"),
            entry_point="main",
            parameters={
                "model_export": 'model:latest',
                "test_data": 'data_test.csv:latest',
            }
        )

if __name__ == '__main__':
    main()
