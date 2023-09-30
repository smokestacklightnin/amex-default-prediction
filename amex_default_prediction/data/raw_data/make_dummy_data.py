import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StringType,
    IntegerType,
)
from importlib.resources import files as get_module_path

default_data_path = get_module_path("amex_default_prediction.data.raw_data")
default_train_data_filename = "train_data_subset.csv"
default_test_data_filename = "test_data_subset.csv"

spark = SparkSession.builder.master("local[10]").getOrCreate()

train_labels_full = (
    spark.read.format("csv")
    .option("header", True)
    .schema(
        StructType()
        .add("customer_ID", StringType(), True)
        .add("target", IntegerType(), True)
    )
    .load("./train_labels.csv")
)

train_data_full = (
    spark.read.format("csv").option("header", True).load("./train_data.csv")
)


def make_train_labels_subset(fraction=0.1):
    return train_labels_full.sample(
        fraction=fraction,
        withReplacement=False,
    )


def make_train_data_subset_from_train_labels(labels):
    return train_data_full.join(
        labels,
        "customer_ID",
        "right",
    )


def write_data_subset(df, filename, filepath=default_data_path):
    df.write.options(
        header=True,
    ).mode(
        "overwrite"
    ).csv(filepath.joinpath(filename).as_posix(), compression="none")


if __name__ == "__main__":
    train_labels_subset = make_train_labels_subset(0.04)
    train_data_subset = make_train_data_subset_from_train_labels(train_labels_subset)
    write_data_subset(train_labels_subset, "train_labels_subset")
    write_data_subset(train_data_subset, "train_data_subset")
