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


def make_train_labels_subset(fraction=0.1):
    return train_labels_full.sample(
        fraction=fraction,
        withReplacement=False,
    )


if __name__ == "__main__":
    labels_subset = make_train_labels_subset(0.02)
