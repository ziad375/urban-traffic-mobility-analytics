from pyspark.sql import SparkSession, functions as F

spark = (
    SparkSession.builder
    .appName("CheckCluster4")
    .getOrCreate()
)

source_path = "hdfs://namenode:8020/data/nyc/processed/zone_hour"
clusters_path = "hdfs://namenode:8020/data/nyc/sparkml/clusters"

source = spark.read.parquet(source_path)

clusters = spark.read.parquet(clusters_path).select(
    "pu_zone", "pickup_date", "pickup_hour", "prediction"
)

df = source.join(
    clusters,
    ["pu_zone", "pickup_date", "pickup_hour"],
    "inner"
)

# 1. Inspect cluster 4's highest-volume zone-hour records
c4 = df.filter(F.col("prediction") == 4)

print("Cluster 4: Top 20 zone-hour records by trip count")

c4.select(
    "pu_zone", "pickup_date", "pickup_hour",
    "trips", "total_revenue", "avg_fare",
    "avg_distance", "avg_duration_min", "avg_speed_mph"
).orderBy(F.desc("trips")).show(20, truncate=False)

print("Cluster 4: Top 20 zone-hour records by revenue")

c4.select(
    "pu_zone", "pickup_date", "pickup_hour",
    "trips", "total_revenue", "avg_fare"
).orderBy(F.desc("total_revenue")).show(20, truncate=False)

# 2. Summarize the distribution to identify extreme values
print("Cluster 4: Metric quantiles")

for col_name in ["trips", "total_revenue", "avg_fare",
                 "avg_distance", "avg_duration_min", "avg_speed_mph"]:
    quantiles = c4.approxQuantile(col_name, [0.50, 0.90, 0.95, 0.99], 0.01)
    print(col_name, "P50, P90, P95, P99 =", quantiles)

# 3. Count zone-hour records with unusually high trip counts
print("Records with trips >= 500:")
c4.filter(F.col("trips") >= 500).count()

print("Count =", c4.filter(F.col("trips") >= 500).count())

spark.stop()