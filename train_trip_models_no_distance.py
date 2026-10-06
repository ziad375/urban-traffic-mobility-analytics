import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb

from pyspark.sql import SparkSession, functions as F

spark = (
    SparkSession.builder
    .appName("XGBoostDurationCostNoDistance")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

SOURCE = "hdfs://namenode:8020/data/nyc/processed/trips"
OUT = "/tmp/xgboost_trip_models_no_distance"
os.makedirs(OUT, exist_ok=True)

print("Reading cleaned trip data...")
df = spark.read.parquet(SOURCE)

required = [
    "pickup_ts", "pu_zone", "do_zone",
    "duration_min", "total_amount"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df = (
    df.withColumn("pickup_ts", F.to_timestamp("pickup_ts"))
      .withColumn("duration_min", F.col("duration_min").cast("double"))
      .withColumn("total_amount", F.col("total_amount").cast("double"))
      .withColumn("pu_zone", F.col("pu_zone").cast("double"))
      .withColumn("do_zone", F.col("do_zone").cast("double"))
      .filter(
          F.col("pickup_ts").isNotNull()
          & F.col("pu_zone").isNotNull()
          & F.col("do_zone").isNotNull()
          & F.col("duration_min").isNotNull()
          & F.col("total_amount").isNotNull()
          & (F.col("duration_min") > 0)
          & (F.col("duration_min") <= 300)
          & (F.col("total_amount") > 0)
          & (F.col("total_amount") <= 500)
      )
      .withColumn("pickup_hour", F.hour("pickup_ts"))
      .withColumn("day_of_week", F.dayofweek("pickup_ts"))
      .withColumn("month", F.month("pickup_ts"))
      .withColumn("year", F.year("pickup_ts"))
      .withColumn(
          "is_weekend",
          F.dayofweek("pickup_ts").isin([1, 7]).cast("int")
      )
)

# Keep the same temporal split as the previous model.
train_spark = df.filter(F.col("pickup_ts") < F.lit("2025-10-01"))
test_spark = df.filter(F.col("pickup_ts") >= F.lit("2025-10-01"))

train_spark = train_spark.sample(False, 0.01, seed=42)
test_spark = test_spark.sample(False, 0.02, seed=43)

# No trip distance and no passenger count.
features = [
    "pu_zone", "do_zone", "pickup_hour",
    "day_of_week", "month", "year", "is_weekend"
]
targets = ["duration_min", "total_amount"]

print("Collecting training sample...")
train = train_spark.select(*(features + targets)).toPandas()

print("Collecting test sample...")
test = test_spark.select(*(features + targets)).toPandas()

print("Training sample rows:", len(train))
print("Test sample rows:", len(test))

if len(train) < 1000 or len(test) < 1000:
    raise RuntimeError("Sample too small; inspect the logs.")

X_train = train[features].astype("float32")
X_test = test[features].astype("float32")

def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype="float64")
    predicted = np.asarray(predicted, dtype="float64")

    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    denom = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - np.sum((actual - predicted) ** 2) / denom if denom else None

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2) if r2 is not None else None
    }

results = {}

for target in targets:
    print(f"\nTraining model for {target}...")

    y_train = train[target].to_numpy(dtype="float64")
    y_test = test[target].to_numpy(dtype="float64")

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=4,
        random_state=42
    )

    model.fit(X_train, y_train, verbose=False)
    predictions = np.maximum(0, model.predict(X_test))

    metrics = calculate_metrics(y_test, predictions)
    model_path = os.path.join(OUT, f"{target}_xgboost.json")
    model.save_model(model_path)

    results[target] = {
        **metrics,
        "train_sample_rows": len(train),
        "test_sample_rows": len(test),
        "features": features,
        "target": target
    }

    print(f"{target} MAE:  {metrics['MAE']:.4f}")
    print(f"{target} RMSE: {metrics['RMSE']:.4f}")
    print(f"{target} R2:   {metrics['R2']}")

with open(os.path.join(OUT, "metrics.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\nFINAL RESULTS")
print(json.dumps(results, indent=2))
print("\nModels saved under:", OUT)

spark.stop()
