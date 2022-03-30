# -*- coding: utf-8 -*-
"""Model component.

Created on: 3/30/2022
@author: Heber Trujillo <heber.trj.urt@gmail.com> 
Licence,
"""
from typing import List, Optional, Tuple
import argparse
import itertools
import logging
import os

import yaml
import tempfile
import mlflow
import pandas as pd
import numpy as np
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, FunctionTransformer

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import wandb
from wandb.sdk.wandb_run import Run

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def main(
        train_data: str,
        model_config: str,
        export_artifact: str,
        random_seed: int,
        val_size: float,
        stratify: Optional[str]
):
    """train and load to W&B an inference artifact.

    Args:
        train_data: Fully-qualified name for the training data artifact
        model_config: Path to a YAML file containing the configuration
            for the random forest.
        export_artifact: Name of the artifact for the exported model. Use
            'null' for no export.
        random_seed: Seed for the random number generator.
        val_size: Size for the validation set as a fraction of the training set.
        stratify: Name of a column to be used for stratified sampling.

    Returns:

    """

    run = wandb.init(job_type="train")

    logger.info("Downloading and reading train artifact")
    train_data_path = run.use_artifact(train_data).file()
    df = pd.read_csv(train_data_path, low_memory=False)

    # Extract the target from the features
    logger.info("Extracting target from dataframe")
    X = df.copy()
    y = X.pop("genre")

    logger.info("Splitting train/val")
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=val_size,
        stratify=df[stratify] if stratify != "null" else None,
        random_state=random_seed,
    )

    logger.info("Setting up pipeline")

    pipe, used_columns = get_training_inference_pipeline(
        model_config=model_config
    )

    logger.info("Fitting")
    pipe.fit(X_train[used_columns], y_train)

    # Evaluate
    pred = pipe.predict(X_val[used_columns])
    pred_proba = pipe.predict_proba(X_val[used_columns])

    logger.info("Scoring")
    score = roc_auc_score(y_val, pred_proba,
                          average="macro", multi_class="ovo")

    run.summary["AUC"] = score

    # Export if required
    if export_artifact != "null":

        export_model(run, pipe, used_columns, X_val, pred, export_artifact)

    # Some useful plots
    fig_feat_imp = plot_feature_importance(pipe)

    fig_cm, sub_cm = plt.subplots(figsize=(10, 10))
    plot_confusion_matrix(
        pipe,
        X_val[used_columns],
        y_val,
        ax=sub_cm,
        normalize="true",
        values_format=".1f",
        xticks_rotation=90,
    )
    fig_cm.tight_layout()

    run.log(
        {
            "feature_importance": wandb.Image(fig_feat_imp),
            "confusion_matrix": wandb.Image(fig_cm),
        }
    )


def export_model(
        run: Run,
        pipe: Pipeline,
        used_columns: List[str],
        X_val: pd.DataFrame,
        val_pred: np.ndarray,
        export_artifact: str
) -> None:
    """Export inference artifact to W&B.

    Args:
        run: W&B run session.
        pipe: Inference artifact.
        used_columns: List of columns used by the inference artifact.
        X_val: Validation set.
        val_pred: Predictions for validation set.
        export_artifact: Name of the artifact for the exported model. Use
            'null' for no export.

    Returns:
        None

    """

    # Infer the signature of the model

    # Get the columns that we are really using from the pipeline
    signature = infer_signature(X_val[used_columns], val_pred)

    with tempfile.TemporaryDirectory() as temp_dir:

        export_path = os.path.join(temp_dir, "model_export")

        mlflow.sklearn.save_model(
            pipe,
            export_path,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            signature=signature,
            input_example=X_val.iloc[:2],
        )

        artifact = wandb.Artifact(
            export_artifact,
            type="model_export",
            description="Random Forest pipeline export",
        )
        artifact.add_dir(export_path)

        run.log_artifact(artifact)

        # Make sure the artifact is uploaded before the temp dir
        # gets deleted
        artifact.wait()


def plot_feature_importance(
        pipe: Pipeline
) -> Figure:
    """Create feature importance plot for the inference artifact.

    Args:
        pipe: Inference artifact.

    Returns:
        fig_feat_imp: feature importance plot.
    """

    # We collect the feature importance for all non-nlp features first
    feat_names = np.array(
        pipe["preprocessor"].transformers[0][-1]
        + pipe["preprocessor"].transformers[1][-1]
    )
    feat_imp = pipe["classifier"].feature_importances_[: len(feat_names)]
    # For the NLP feature we sum across all the TF-IDF dimensions into a global
    # NLP importance
    nlp_importance = sum(pipe["classifier"].feature_importances_[len(feat_names) :])
    feat_imp = np.append(feat_imp, nlp_importance)
    feat_names = np.append(feat_names, "title + song_name")
    fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10))
    idx = np.argsort(feat_imp)[::-1]
    sub_feat_imp.bar(range(feat_imp.shape[0]), feat_imp[idx], color="r", align="center")
    _ = sub_feat_imp.set_xticks(range(feat_imp.shape[0]))
    _ = sub_feat_imp.set_xticklabels(feat_names[idx], rotation=90)
    fig_feat_imp.tight_layout()
    return fig_feat_imp


def get_training_inference_pipeline(
        model_config: str,
) -> Tuple[Pipeline, List[str]]:
    """Creat an inference artifact.

    Args:
        model_config: Path to a YAML file containing the configuration for
            the random forest.

    Returns:
        pipe: Inference artifact.
        used_columns: List of column names used by the inference artifact.

    """

    # Get the configuration for the pipeline
    with open(model_config) as fp:
        model_config = yaml.safe_load(fp)
    # Add it to the W&B configuration so the values for the hyperparams
    # are tracked
    wandb.config.update(model_config)

    # We need 3 separate preprocessing "tracks":
    # - one for categorical features
    # - one for numerical features
    # - one for textual ("nlp") features
    # Categorical preprocessing pipeline
    categorical_features = sorted(model_config["features"]["categorical"])
    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0), OrdinalEncoder()
    )
    # Numerical preprocessing pipeline
    numeric_features = sorted(model_config["features"]["numerical"])
    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler()
    )
    # Textual ("nlp") preprocessing pipeline
    nlp_features = sorted(model_config["features"]["nlp"])
    # This trick is needed because SimpleImputer wants a 2d input, but
    # TfidfVectorizer wants a 1d input. So we reshape in between the two steps
    reshape_to_1d = FunctionTransformer(np.reshape, kw_args={"newshape": -1})
    nlp_transformer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=""),
        reshape_to_1d,
        TfidfVectorizer(
            binary=True, max_features=model_config["tfidf"]["max_features"]
        ),
    )
    # Put the 3 tracks together into one pipeline using the ColumnTransformer
    # This also drops the columns that we are not explicitly transforming
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
            ("nlp1", nlp_transformer, nlp_features),
        ],
        remainder="drop",  # This drops the columns that we do not transform
    )

    # Get a list of the columns we used
    used_columns = list(
        itertools.chain.from_iterable(
            [x[2] for x in preprocessor.transformers]
        )
    )

    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(**model_config["random_forest"])),
        ]
    )
    return pipe, used_columns


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Random Forest",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--train_data",
        type=str,
        help="Fully-qualified name for the training data artifact",
        required=True,
    )

    parser.add_argument(
        "--model_config",
        type=str,
        help="Path to a YAML file containing the configuration for the random forest",
        required=True,
    )

    parser.add_argument(
        "--export_artifact",
        type=str,
        help="Name of the artifact for the exported model. Use 'null' for no export.",
        required=False,
        default="null",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Seed for the random number generator.",
        required=False,
        default=42
    )

    parser.add_argument(
        "--val_size",
        type=float,
        help="Size for the validation set as a fraction of the training set",
        required=False,
        default=0.3
    )

    parser.add_argument(
        "--stratify",
        type=str,
        help="Name of a column to be used for stratified sampling. Default: "
             "'null', i.e., no stratification",
        required=False,
        default="null",
    )

    args = parser.parse_args()

    main(
        train_data=args.train_data,
        model_config=args.model_config,
        export_artifact=args.export_artifact,
        random_seed=args.random_seed,
        val_size=args.val_size,
        stratify=args.stratify
    )
