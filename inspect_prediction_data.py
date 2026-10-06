from pyspark.sql import SparkSession
import importlib.util

spark = SparkSession.builder.appName("InspectPredictionData").getOrCreate()

paths = {
    "zone_hour": "hdfs://namenode:8020/data/nyc/processed/zone_hour",
    "trips": "hdfs://namenode:8020/data/nyc/processed/trips"
}

for name, path in paths.items():
    print("\n" + "=" * 60)
    print(f"DATASET: {name}")
    print("=" * 60)

    try:
        df = spark.read.parquet(path)
        print("Columns:", df.columns)
        df.printSchema()
        print("Sample rows:")
        df.show(3, truncate=False)
    except Exception as e:
        print(f"ERROR reading {name}: {str(e)[:1500]}")

print("\n" + "=" * 60)
print("XGBoost availability")
print("=" * 60)
print("xgboost installed:", importlib.util.find_spec("xgboost") is not None)

spark.stop()
