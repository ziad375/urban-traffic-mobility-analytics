"""
Train XGBoost models that predict, for a single taxi trip:
  - duration_min  (trip duration in minutes)
  - total_amount  (total fare in USD)

Input : cleaned trips written by the ETL -> hdfs://namenode:8020/data/nyc/processed/trips
Split : temporal holdout (train < 2025-10-01 <= test), both sets randomly sampled
        so the data fits in the driver's memory.

Two variants (this file replaces train_trip_models.py + train_trip_models_no_distance.py,
which are kept in original/):
  default        features include trip_distance + passenger_count
                 -> saved to /tmp/xgboost_trip_models            (copy to models/trip_models)
                    this is the model the Streamlit app uses
  --no-distance  same models WITHOUT trip_distance / passenger_count
                 -> saved to /tmp/xgboost_trip_models_no_distance (copy to models/trip_models_no_distance)

Run (inside the Spark container; needs xgboost, pandas and numpy installed there):
  spark-submit --master spark://spark-master:7077 train_trip_models.py [--no-distance]
"""
import argparse
import json
import os
from functools import reduce

import numpy as np
import xgboost as xgb
from pyspark.sql import SparkSession, functions as F

# ---------- Variant selection ----------
parser = argparse.ArgumentParser(description="Train trip duration / fare XGBoost models.")
parser.add_argument(
    "--no-distance",
    action="store_true",
    help="Train without trip_distance and passenger_count.",
)
args = parser.parse_args()
use_distance = not args.no_distance

if use_distance:
    APP_NAME = "XGBoostDurationCost"
    OUT = "/tmp/xgboost_trip_models"
    features = [
        "pu_zone", "do_zone", "passenger_count", "trip_distance",
        "pickup_hour", "day_of_week", "month", "year", "is_weekend",
    ]
else:
    APP_NAME = "XGBoostDurationCostNoDistance"
    OUT = "/tmp/xgboost_trip_models_no_distance"
    features = [
        "pu_zone", "do_zone", "pickup_hour",
        "day_of_week", "month", "year", "is_weekend",
    ]

targets = ["duration_min", "total_amount"]

spark = SparkSession.builder.appName(APP_NAME).getOrCreate()
spark.sparkContext.setLogLevel("WARN")

SOURCE = "hdfs://namenode:8020/data/nyc/processed/trips"
os.makedirs(OUT, exist_ok=True)

# ---------- 1) Read cleaned trips ----------
print("Reading cleaned trip data...")
df = spark.read.parquet(SOURCE)

required = ["pickup_ts", "pu_zone", "do_zone", "duration_min", "total_amount"]
if use_distance:
    required += ["passenger_count", "trip_distance"]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# ---------- 2) Types, filtering, calendar features ----------
numeric_cols = ["duration_min", "total_amount", "pu_zone", "do_zone"]
if use_distance:
    numeric_cols += ["trip_distance", "passenger_count"]

df = df.withColumn("pickup_ts", F.to_timestamp("pickup_ts"))
for c in numeric_cols:
    df = df.withColumn(c, F.col(c).cast("double"))

# Drop nulls and extreme values (stricter than the ETL, to keep training stable).
conditions = [
    F.col("pickup_ts").isNotNull(),
    F.col("pu_zone").isNotNull(),
    F.col("do_zone").isNotNull(),
    F.col("duration_min").isNotNull(),
    F.col("total_amount").isNotNull(),
    F.col("duration_min") > 0,
    F.col("duration_min") <= 300,
    F.col("total_amount") > 0,
    F.col("total_amount") <= 500,
]
if use_distance:
    conditions += [
        F.col("passenger_count").isNotNull(),
        F.col("trip_distance").isNotNull(),
        F.col("passenger_count") >= 0,
        F.col("trip_distance") > 0,
        F.col("trip_distance") <= 100,
    ]
df = df.filter(reduce(lambda a, b: a & b, conditions))

df = (
    df.withColumn("pickup_hour", F.hour("pickup_ts"))
      .withColumn("day_of_week", F.dayofweek("pickup_ts"))  # Spark: Sunday=1 ... Saturday=7
      .withColumn("month", F.month("pickup_ts"))
      .withColumn("year", F.year("pickup_ts"))
      .withColumn("is_weekend", F.dayofweek("pickup_ts").isin([1, 7]).cast("int"))
)

# ---------- 3) Temporal train/test split + sampling ----------
# Train on trips before Oct 2025, test from Oct 2025 onward.
train_spark = df.filter(F.col("pickup_ts") < F.lit("2025-10-01"))
test_spark = df.filter(F.col("pickup_ts") >= F.lit("2025-10-01"))

# Sampling keeps local XGBoost training manageable; it happens after filtering
# and does not modify the HDFS source data.
train_spark = train_spark.sample(False, 0.01, seed=42)
test_spark = test_spark.sample(False, 0.02, seed=43)

print("Collecting training sample...")
train = train_spark.select(*(features + targets)).toPandas()

print("Collecting test sample...")
test = test_spark.select(*(features + targets)).toPandas()

print("Training sample rows:", len(train))
print("Test sample rows:", len(test))

if len(train) < 1000 or len(test) < 1000:
    raise RuntimeError("Sample too small; inspect the logs before proceeding.")

X_train = train[features].astype("float32")
X_test = test[features].astype("float32")


def calculate_metrics(actual, predicted):
    """Return MAE, RMSE and R2 for one set of predictions."""
    actual = np.asarray(actual, dtype="float64")
    predicted = np.asarray(predicted, dtype="float64")

    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    denom = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - np.sum((actual - predicted) ** 2) / denom if denom else None

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2) if r2 is not None else None,
    }


# ---------- 4) Train one model per target, evaluate, save ----------
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
        random_state=42,
    )

    model.fit(X_train, y_train, verbose=False)
    predictions = np.maximum(0, model.predict(X_test))  # durations/fares can't be negative

    metrics = calculate_metrics(y_test, predictions)
    model.save_model(os.path.join(OUT, f"{target}_xgboost.json"))

    results[target] = {
        **metrics,
        "train_sample_rows": len(train),
        "test_sample_rows": len(test),
        "features": features,
        "target": target,
    }

    print(f"{target} MAE:  {metrics['MAE']:.4f}")
    print(f"{target} RMSE: {metrics['RMSE']:.4f}")
    print(f"{target} R2:   {metrics['R2']}")

with open(os.path.join(OUT, "metrics.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\nRESULTS")
print(json.dumps(results, indent=2))
print("\nModels saved under:", OUT)

spark.stop()
