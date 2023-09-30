import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StringType,
    IntegerType,
)

spark = SparkSession.builder.master("local[8]").getOrCreate()

train_labels_full = (
    spark.read.format("csv")
    .option("header", True)
    .schema(
        StructType()
        .add("customer_ID", StringType(), True)
        .add("defaulted", IntegerType(), True)
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
        labels_subset,
        "customer_ID",
        "right",
    )


if __name__ == "__main__":
    labels_subset = make_train_labels_subset(0.02)
    data_subset = make_train_data_subset_from_train_labels(labels_subset)
