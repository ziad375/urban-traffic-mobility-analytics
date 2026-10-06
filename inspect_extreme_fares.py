from pyspark.sql import SparkSession, functions as F

spark = (
    SparkSession.builder
    .appName("InspectExtremeFares")
    .getOrCreate()
)

path = "hdfs://namenode:8020/data/nyc/processed/trips"
df = spark.read.parquet(path)

print("=== Schema ===")
df.select(
    "pu_zone", "pickup_ts", "dropoff_ts",
    "trip_distance", "duration_min",
    "fare_amount", "total_amount"
).printSchema()

print("=== Highest total_amount ===")
df.select(
    "pu_zone", "pickup_ts", "dropoff_ts",
    "trip_distance", "duration_min",
    "fare_amount", "tip_amount", "total_amount"
).orderBy(F.desc("total_amount")).show(30, truncate=False)

print("=== Highest fare_amount ===")
df.select(
    "pu_zone", "pickup_ts", "dropoff_ts",
    "trip_distance", "duration_min",
    "fare_amount", "tip_amount", "total_amount"
).orderBy(F.desc("fare_amount")).show(30, truncate=False)

print("=== Numeric quantiles ===")
for col_name in ["fare_amount", "total_amount", "trip_distance", "duration_min"]:
    values = df.approxQuantile(
        col_name, [0.50, 0.90, 0.99, 0.999], 0.001
    )
    print(col_name, "P50, P90, P99, P99.9 =", values)

spark.stop()
