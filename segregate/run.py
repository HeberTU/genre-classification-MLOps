# -*- coding: utf-8 -*-
"""Data Segregation component.

Created on: 3/30/2022
@author: Heber Trujillo <heber.trj.urt@gmail.com> 
Licence,
"""
import argparse
import logging
import os
import tempfile

import pandas as pd
import wandb
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def main(
        input_artifact: str,
        artifact_root: str,
        artifact_type: str,
        test_size: float,
        random_state: int,
        stratify: str
) -> None:
    """Split the input artifact into train and test sets.

    Args:
        input_artifact: Fully-qualified name for the input artifact.
        artifact_root: Root for the names of the produced artifacts. The script
            will produce 2 artifacts: "{root}_train.csv and {root}_test.csv
        artifact_type: Type for the produced artifacts.
        test_size: Fraction of dataset or number of items to include in the test split.
        random_state: An integer number to use to init the random number generator.
        stratify: If set, it is the name of a column to use for stratified splitting

    Returns:

    """

    run = wandb.init(job_type="split_data")

    logger.info("Downloading and reading artifact")
    artifact = run.use_artifact(input_artifact)
    artifact_path = artifact.file()

    df = pd.read_csv(artifact_path, low_memory=False)

    # Split first in model_dev/test, then we further divide model_dev in train and validation
    logger.info("Splitting data into train, val and test")
    splits = {}

    splits["train"], splits["test"] = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[stratify] if stratify != 'null' else None,
    )

    # Save the artifacts. We use a temporary directory so we do not leave
    # any trace behind
    with tempfile.TemporaryDirectory() as tmp_dir:

        for split, df in splits.items():

            # Make the artifact name from the provided root plus the name of the split
            artifact_name = f"{artifact_root}_{split}.csv"

            # Get the path on disk within the temp directory
            temp_path = os.path.join(tmp_dir, artifact_name)

            logger.info(f"Uploading the {split} dataset to {artifact_name}")

            # Save then upload to W&B
            df.to_csv(temp_path)

            artifact = wandb.Artifact(
                name=artifact_name,
                type=artifact_type,
                description=f"{split} split of dataset {input_artifact}",
            )
            artifact.add_file(temp_path)

            logger.info("Logging artifact")
            run.log_artifact(artifact)

            # This waits for the artifact to be uploaded to W&B. If you
            # do not add this, the temp directory might be removed before
            # W&B had a chance to upload the datasets, and the upload
            # might fail
            artifact.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a dataset into train and test",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--artifact_root",
        type=str,
        help="Root for the names of the produced artifacts. The script will "
             "produce 2 artifacts: {root}_train.csv and {root}_test.csv",
        required=True,
    )

    parser.add_argument(
        "--artifact_type", type=str, help="Type for the produced artifacts",
        required=True
    )

    parser.add_argument(
        "--test_size",
        help="Fraction of dataset or number of items to include in the test split",
        type=float,
        required=True
    )

    parser.add_argument(
        "--random_state",
        help="An integer number to use to init the random number generator."
             " It ensures repeatibility in the"
             "splitting",
        type=int,
        required=False,
        default=42
    )

    parser.add_argument(
        "--stratify",
        help="If set, it is the name of a column to use for stratified splitting",
        type=str,
        required=False,
        default='null'  # unfortunately mlflow does not support well optional parameters
    )

    args = parser.parse_args()

    main(
        input_artifact=args.input_artifact,
        artifact_root=args.artifact_root,
        artifact_type=args.artifact_type,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=args.stratify
    )