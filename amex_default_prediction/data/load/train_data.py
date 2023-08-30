import pandas as pd
from amex_default_prediction.data.raw_data.download import _default_output_path
from numba import vectorize
import tensorflow as tf
import datetime

_train_labels = pd.read_csv(
    _default_output_path.joinpath("train_labels.csv"),
    index_col="customer_ID",
    dtype={
        "customer_ID": str,
        "target": int,
    },
)


# def train_data(batch_size=10, shuffle_seed=None):
#     dataset = tf.data.experimental.make_csv_dataset(
#         _default_output_path.joinpath("train_data.csv").as_posix(),
#         batch_size=batch_size,
#         label_name="customer_ID",
#         shuffle=True,
#         shuffle_seed=None,
#     )

#     dataset.map(
#         lambda x, *args: print(
#             datetime.datetime.fromisoformat(str(x["S_2"][0])).toordinal(),
#             *args,
#         )
#     )
#     return dataset


def train_labels(customer_ID):
    return _train_labels.loc[customer_ID, "target"]


if __name__ == "__main__":
    print(_train_labels)
    print(
        train_labels(
            10
            * [
                str("0000099d6bd597052cdcda90ffabf56573fe9d7c79be5fbac11a8ed792feb62a"),
            ],
        )
    )

    print(
        train_labels("0000099d6bd597052cdcda90ffabf56573fe9d7c79be5fbac11a8ed792feb62a")
    )

    # print(train_data().take(1))
