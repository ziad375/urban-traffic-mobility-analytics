
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from functools import reduce

spark = (
    SparkSession.builder
    .appName("nyc-taxi-etl-clean")
    .master("spark://spark-master:7077")
    .getOrCreate()
)

RAW = "hdfs://namenode:8020/data/nyc/raw/"
PROCESSED = "hdfs://namenode:8020/data/nyc/processed/"


# ---------- 1) Read ----------
# Read each Parquet file separately to handle schema differences.


hadoop_conf = spark._jsc.hadoopConfiguration()
raw_path = spark._jvm.org.apache.hadoop.fs.Path(RAW)
fs = raw_path.getFileSystem(hadoop_conf)

files = [
    status.getPath().toString()
    for status in fs.listStatus(raw_path)
    if status.isFile()
    and status.getPath().getName().endswith(".parquet")
]


if not files:
    raise RuntimeError(f"No Parquet files found in {RAW}")

dataframes = []

for file_path in files:
    file_df = spark.read.parquet(file_path)

    # Normalize numeric columns before combining files.
    for column in [
        "VendorID",
        "passenger_count",
        "PULocationID",
        "DOLocationID",
        "payment_type",
    ]:
        if column in file_df.columns:
            file_df = file_df.withColumn(
                column, F.col(column).cast("int")
            )

    for column in [
        "trip_distance",
        "fare_amount",
        "tip_amount",
        "total_amount",
    ]:
        if column in file_df.columns:
            file_df = file_df.withColumn(
                column, F.col(column).cast("double")
            )

    dataframes.append(file_df)

raw = reduce(
    lambda left, right: left.unionByName(
        right, allowMissingColumns=True
    ),
    dataframes
)

n_raw = raw.count()

df = raw.select(
    F.col("VendorID").cast("int").alias("vendor_id"),
    F.col("tpep_pickup_datetime").cast("timestamp").alias("pickup_ts"),
    F.col("tpep_dropoff_datetime").cast("timestamp").alias("dropoff_ts"),
    F.col("passenger_count").cast("int").alias("passenger_count"),
    F.col("trip_distance").cast("double").alias("trip_distance"),
    F.col("PULocationID").cast("int").alias("pu_zone"),
    F.col("DOLocationID").cast("int").alias("do_zone"),
    F.col("payment_type").cast("int").alias("payment_type"),
    F.col("fare_amount").cast("double").alias("fare_amount"),
    F.col("tip_amount").cast("double").alias("tip_amount"),
    F.col("total_amount").cast("double").alias("total_amount"),
)


# ---------- 2) Feature Engineering ----------
df = (
    df
    .withColumn(
        "duration_min",
        (F.unix_timestamp("dropoff_ts")
         - F.unix_timestamp("pickup_ts")) / 60.0
    )
    .withColumn("pickup_date", F.to_date("pickup_ts"))
    .withColumn("pickup_hour", F.hour("pickup_ts"))
    .withColumn("day_of_week", F.dayofweek("pickup_ts"))
    .withColumn("year", F.year("pickup_ts"))
    .withColumn("month", F.month("pickup_ts"))
    .withColumn(
        "avg_speed_mph",
        F.col("trip_distance") / (F.col("duration_min") / 60.0)
    )
)

# ---------- 3) Data Quality Report ----------
# Define valid conditions. NULL values are handled as invalid.
validity = {
    "Missing pickup/dropoff zones":
        F.col("pu_zone").isNotNull() &
        F.col("do_zone").isNotNull(),

    "Missing timestamps":
        F.col("pickup_ts").isNotNull() &
        F.col("dropoff_ts").isNotNull(),

    "Invalid year":
        F.col("year").between(2021, 2026),

    "Invalid duration":
        F.col("duration_min").between(1, 180),

    "Invalid distance":
        (F.col("trip_distance") > 0) &
        (F.col("trip_distance") <= 100),

    "Invalid fare":
        (F.col("fare_amount") > 0) &
        (F.col("total_amount") > 0),

    "Invalid passenger count":
        F.col("passenger_count").between(1, 6),

    "Invalid speed":
        F.col("avg_speed_mph") <= 80,
}

# A NULL validity result is treated as a failed rule.
invalid = {
    name: ~F.coalesce(condition, F.lit(False))
    for name, condition in validity.items()
}

# Count failed rows for every rule.
# Individual counts can overlap.
report_expressions = [
    F.sum(F.when(condition, 1).otherwise(0)).alias(name)
    for name, condition in invalid.items()
]

# Count how many rules each row fails.
failed_rules_count = reduce(
    lambda a, b: a.cast("int") + b.cast("int"),
    list(invalid.values())
)

any_failure = reduce(
    lambda a, b: a | b,
    list(invalid.values())
)

report_expressions.extend([
    F.sum(
        F.when(any_failure, 1).otherwise(0)
    ).alias("Rows failing at least one rule"),

    F.sum(
        F.when(failed_rules_count >= 2, 1).otherwise(0)
    ).alias("Rows failing at least two rules"),
])

print("\n========== DATA QUALITY REPORT ==========")
df.agg(*report_expressions).show(truncate=False)

print("\n========== PASSENGER COUNT DISTRIBUTION ==========")

df.groupBy("passenger_count") \
    .count() \
    .orderBy("passenger_count") \
    .show(50, truncate=False)

# ---------- 4) Cleaning ----------
# Keep the original cleaning rules unchanged.
clean = df.filter(
    F.col("pu_zone").isNotNull()
    & F.col("do_zone").isNotNull()
    & F.col("pickup_ts").isNotNull()
    & F.col("dropoff_ts").isNotNull()
    & F.col("year").between(2021, 2026)
    & (F.col("duration_min") >= 1)
    & (F.col("duration_min") <= 180)
    & (F.col("trip_distance") > 0)
    & (F.col("trip_distance") <= 100)
    & (F.col("fare_amount") > 0)
    & (F.col("total_amount") > 0)
    & F.col("passenger_count").between(1, 6)
    & (F.col("avg_speed_mph") <= 80)
)

# ---------- 5) Write Cleaned Trips ----------
(
    clean.write
    .mode("overwrite")
    .partitionBy("year", "month")
    .parquet(PROCESSED + "trips")
)

# ---------- 6) Aggregation: Zone / Date / Hour ----------
zone_hour = (
    clean.groupBy("pu_zone", "pickup_date", "pickup_hour")
    .agg(
        F.count("*").alias("trips"),
        F.round(F.avg("fare_amount"), 2).alias("avg_fare"),
        F.round(F.avg("trip_distance"), 2).alias("avg_distance"),
        F.round(F.avg("duration_min"), 2).alias("avg_duration_min"),
        F.round(F.avg("avg_speed_mph"), 2).alias("avg_speed_mph"),
        F.round(F.sum("total_amount"), 2).alias("total_revenue"),
    )
)

zone_hour.write.mode("overwrite").parquet(
    PROCESSED + "zone_hour"
)

# ---------- 7) Final Report ----------
n_clean = clean.count()
n_removed = n_raw - n_clean

print("\n" + "=" * 50)
print("ETL SUMMARY")
print("=" * 50)
print("Raw rows      :", n_raw)
print("Clean rows    :", n_clean)
print("Removed rows  :", n_removed)
print("Removal rate  :", f"{n_removed / n_raw:.2%}")
print("Zone-hour rows:", zone_hour.count())
print("=" * 50)

print("\nTop 5 busiest zone-hour groups:")
zone_hour.orderBy(F.desc("trips")).show(5, truncate=False)

spark.stop()
