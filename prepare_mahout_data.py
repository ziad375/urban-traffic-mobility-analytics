from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler

spark = SparkSession.builder \
    .appName("PrepareMahoutClusteringData") \
    .getOrCreate()

source = "hdfs://namenode:8020/data/nyc/processed/zone_hour"
output = "hdfs://namenode:8020/data/nyc/mahout/features"

df = spark.read.parquet(source)

feature_cols = [
    "trips",
    "avg_fare",
    "avg_distance",
    "avg_duration_min",
    "avg_speed_mph",
    "total_revenue"
]

# Keep valid rows for clustering
df = df.dropna(subset=feature_cols)

for col_name in feature_cols:
    df = df.withColumn(
        col_name,
        F.col(col_name).cast("double")
    )

# Assemble numerical features into one vector
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="raw_features"
)

assembled = assembler.transform(df)

# Standardize features
scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withMean=True,
    withStd=True
)

scaler_model = scaler.fit(assembled)
scaled = scaler_model.transform(assembled)

# Save identifiers and standardized features
result = scaled.select(
    "pu_zone",
    "pickup_date",
    "pickup_hour",
    "features"
)

result.write.mode("overwrite").parquet(output)

print("Prepared feature schema:")
result.printSchema()

print("Sample prepared rows:")
result.show(5, truncate=False)

print("Saved features to:", output)

spark.stop()
