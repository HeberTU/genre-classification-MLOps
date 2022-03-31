# -*- coding: utf-8 -*-
"""Model evaluation component.

Created on: 3/31/2022
@author: Heber Trujillo <heber.trj.urt@gmail.com> 
Licence,
"""
import argparse
import itertools
import logging
import pandas as pd
import wandb
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, plot_confusion_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def main(
        model_export: str,
        test_data: str
) -> None:
    run = wandb.init(job_type="test")

    logger.info("Downloading and reading test artifact")
    test_data_path = run.use_artifact(test_data).file()
    df = pd.read_csv(
        filepath_or_buffer=test_data_path,
        low_memory=False
    )

    logger.info("Extracting target from dataframe")
    X_test = df.copy()
    y_test = X_test.pop("genre")

    logger.info("Downloading and reading the exported model")
    model_export_path = run.use_artifact(model_export).download()
    pipe = mlflow.sklearn.load_model(
        model_uri=model_export_path
    )

    logger.info("Generating predictions")
    used_columns = list(
        itertools.chain.from_iterable(
            [x[2] for x in pipe['preprocessor'].transformers]
        )
    )
    pred_proba = pipe.predict_proba(X_test[used_columns])

    logger.info("Scoring")
    score = roc_auc_score(
        y_true=y_test,
        y_score=pred_proba,
        average='macro',
        multi_class='ovo',
    )
    run.summary["AUC"] = score

    logger.info("Computing confussion matrix")
    fig_cm, sub_cm = plt.subplots(figsize=(10, 10))
    plot_confusion_matrix(
        estimator=pipe,
        X=X_test[used_columns],
        y_true=y_test,
        ax=sub_cm,
        normalize="true",
        values_format=".1f",
        xticks_rotation=90,
    )
    fig_cm.tight_layout()

    run.log(
        data={
            "confusion_matrix": wandb.Image(fig_cm)
        }
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Test the provided model on the test artifact",
        fromfile_prefix_chars="@"
    )

    parser.add_argument(
        "--model_export",
        type=str,
        help="Fully-qualified artifact name for the exported model to evaluate.",
        required=True,
    )

    parser.add_argument(
        "--test_data",
        type=str,
        help="Fully-qualified artifact name for the test data",
        required=True,
    )

    args = parser.parse_args()

    main(
        model_export=args.model_export,
        test_data=args.test_data
    )
