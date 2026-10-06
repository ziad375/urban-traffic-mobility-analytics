from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("InspectZoneHourSchema") \
    .getOrCreate()

df = spark.read.parquet(
    "hdfs://namenode:8020/data/nyc/processed/zone_hour"
)

df.printSchema()
df.show(5, truncate=False)

spark.stop()
