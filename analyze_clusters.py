from pyspark.sql import SparkSession, functions as F

spark = (
    SparkSession.builder
    .appName("AnalyzeNYCTaxiClusters")
    .getOrCreate()
)

features_path = "hdfs://namenode:8020/data/nyc/mahout/features"
clusters_path = "hdfs://namenode:8020/data/nyc/sparkml/clusters"
source_path = "hdfs://namenode:8020/data/nyc/processed/zone_hour"

features = spark.read.parquet(features_path)
clusters = spark.read.parquet(clusters_path)
source = spark.read.parquet(source_path)

# Join cluster predictions to the original, unscaled metrics.
# The feature preparation and clustering preserve these identifying columns.
original = source.select(
    "pu_zone",
    "pickup_date",
    "pickup_hour",
    "trips",
    "avg_fare",
    "avg_distance",
    "avg_duration_min",
    "avg_speed_mph",
    "total_revenue"
)

result = clusters.select(
    "pu_zone", "pickup_date", "pickup_hour", "prediction"
).join(
    original,
    ["pu_zone", "pickup_date", "pickup_hour"],
    "inner"
)

print("Original metrics by cluster:")

result.groupBy("prediction").agg(
    F.count("*").alias("records"),
    F.round(F.avg("trips"), 2).alias("avg_trips"),
    F.round(F.avg("avg_fare"), 2).alias("avg_fare"),
    F.round(F.avg("avg_distance"), 2).alias("avg_distance"),
    F.round(F.avg("avg_duration_min"), 2).alias("avg_duration_min"),
    F.round(F.avg("avg_speed_mph"), 2).alias("avg_speed_mph"),
    F.round(F.avg("total_revenue"), 2).alias("avg_revenue")
).orderBy("prediction").show(truncate=False)

spark.stop()