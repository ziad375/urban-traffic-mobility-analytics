from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("InspectNYCTaxiClusters")
    .getOrCreate()
)

path = "hdfs://namenode:8020/data/nyc/sparkml/clusters"

df = spark.read.parquet(path)

print("Cluster distribution:")

(
    df.groupBy("prediction")
    .agg(
        F.count("*").alias("records"),
        F.countDistinct("pu_zone").alias("zones"),
        F.countDistinct("pickup_date").alias("days")
    )
    .orderBy("prediction")
    .show(truncate=False)
)

print("Sample predictions:")

df.select(
    "pu_zone",
    "pickup_date",
    "pickup_hour",
    "prediction"
).show(10, truncate=False)

spark.stop()