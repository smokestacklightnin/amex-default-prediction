import pandas as pd
from amex_default_prediction.data.raw_data.download import _default_output_path

# from numba import vectorize
import tensorflow as tf
import datetime
from pathlib import Path
from logging import warn

if _default_output_path.joinpath("train_labels.csv").is_file():
    _train_labels = pd.read_csv(
        _default_output_path.joinpath("train_labels.csv"),
        index_col="customer_ID",
        dtype={
            "customer_ID": str,
            "target": int,
        },
    )
else:
    warn("Full dataset csv file not found")

_train_labels_subset = pd.concat(
    (
        pd.read_csv(
            f,
            index_col="customer_ID",
            dtype={
                "customer_ID": str,
                "target": int,
            },
        )
        for f in _default_output_path.joinpath("train_labels_subset").glob("*.csv")
    )
)


def train_data(batch_size=10, shuffle_seed=None):
    dataset = tf.data.experimental.make_csv_dataset(
        _default_output_path.joinpath("train_data.csv").as_posix(),
        batch_size=batch_size,
        # label_name="customer_ID", # There are multiple rows with the same `customer_ID`
        shuffle=True,
        shuffle_seed=None,
    )

    return dataset


def train_data_subset(batch_size=10, shuffle_seed=None):
    dataset = tf.data.experimental.make_csv_dataset(
        map(
            lambda x: x.as_posix(),
            _default_output_path.joinpath("train_data_subset").glob("*.csv"),
        ),
        batch_size=batch_size,
        # label_name="customer_ID", # There are multiple rows with the same `customer_ID`
        shuffle=True,
        shuffle_seed=None,
    )

    return dataset


def train_labels(customer_ID):
    return _train_labels.loc[customer_ID, "target"]


def train_labels_subset(customer_ID):
    return _train_labels_subset.loc[customer_ID, "target"]


if __name__ == "__main__":
    print(_train_labels)
    print(
        train_labels(
            10
            * [
                str("00c617b58ce8c94b52a858c4886a7e4736a2373b6e841726421555f10e186763"),
            ],
        )
    )

    print(
        train_labels("00c617b58ce8c94b52a858c4886a7e4736a2373b6e841726421555f10e186763")
    )

    print(train_data_subset().take(1))
