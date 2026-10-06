from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("XGBoostDataPreparation").getOrCreate()

df = spark.read.parquet("hdfs://namenode:8020/data/nyc/processed/zone_hour")

print("Columns:", df.columns)
print("Preparing chronological training and test datasets...")

df = df.withColumn("pickup_date", F.to_date("pickup_date"))
df = df.filter(
    F.col("pu_zone").isNotNull()
    & F.col("pickup_date").isNotNull()
    & F.col("pickup_hour").isNotNull()
    & F.col("trips").isNotNull()
    & F.col("total_revenue").isNotNull()
)

df = (
    df.withColumn("day_of_week", F.dayofweek("pickup_date"))
      .withColumn("day_of_month", F.dayofmonth("pickup_date"))
      .withColumn("month", F.month("pickup_date"))
      .withColumn("year", F.year("pickup_date"))
      .withColumn("date_number", F.datediff("pickup_date", F.lit("2022-01-01")))
)

train = df.filter(F.col("pickup_date") < F.lit("2025-10-01"))
test = df.filter(F.col("pickup_date") >= F.lit("2025-10-01"))

print("Training date range: before 2025-10-01")
print("Test date range: 2025-10-01 onward")
print("Feature columns:", [
    "pu_zone", "pickup_hour", "day_of_week",
    "day_of_month", "month", "year", "date_number"
])

print("Train rows:", train.count())
print("Test rows:", test.count())

print("Training sample:")
train.select(
    "pu_zone", "pickup_date", "pickup_hour",
    "trips", "total_revenue"
).orderBy(F.desc("pickup_date")).show(5, truncate=False)

print("Test sample:")
test.select(
    "pu_zone", "pickup_date", "pickup_hour",
    "trips", "total_revenue"
).orderBy("pickup_date").show(5, truncate=False)

spark.stop()
