import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

spark = (
    SparkSession.builder
    .appName("XGBoostTripsRevenue")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

SOURCE = "hdfs://namenode:8020/data/nyc/processed/zone_hour"
OUT = "/tmp/xgboost_models"
os.makedirs(OUT, exist_ok=True)

print("Reading zone_hour...")
df = spark.read.parquet(SOURCE)

df = (
    df.withColumn("pickup_date", F.to_date("pickup_date"))
      .filter(
          F.col("pu_zone").isNotNull()
          & F.col("pickup_date").isNotNull()
          & F.col("pickup_hour").isNotNull()
          & F.col("trips").isNotNull()
          & F.col("total_revenue").isNotNull()
          & (F.col("trips") >= 0)
          & (F.col("total_revenue") >= 0)
      )
      .withColumn(
          "ts",
          F.to_timestamp(
              F.concat(
                  F.col("pickup_date").cast("string"),
                  F.lit(" "),
                  F.lpad(F.col("pickup_hour").cast("string"), 2, "0"),
                  F.lit(":00:00")
              ),
              "yyyy-MM-dd HH:mm:ss"
          )
      )
      .withColumn("day_of_week", F.dayofweek("pickup_date"))
      .withColumn("day_of_month", F.dayofmonth("pickup_date"))
      .withColumn("month", F.month("pickup_date"))
      .withColumn("year", F.year("pickup_date"))
      .withColumn(
          "date_number",
          F.datediff("pickup_date", F.lit("2022-01-01"))
      )
)

w = Window.partitionBy("pu_zone").orderBy("ts")

df = (
    df.withColumn("prev_trips", F.lag("trips", 1).over(w))
      .withColumn("prev_revenue", F.lag("total_revenue", 1).over(w))
      .withColumn("prev_ts", F.lag("ts", 1).over(w))
      .withColumn(
          "hours_since_prev",
          (
              F.col("ts").cast("long") -
              F.col("prev_ts").cast("long")
          ) / 3600.0
      )
      .filter(
          F.col("prev_trips").isNotNull()
          & F.col("prev_revenue").isNotNull()
          & F.col("hours_since_prev").isNotNull()
          & (F.col("hours_since_prev") > 0)
      )
)

features = [
    "pu_zone", "pickup_hour", "day_of_week",
    "day_of_month", "month", "year", "date_number",
    "prev_trips", "prev_revenue", "hours_since_prev"
]

train_spark = df.filter(F.col("pickup_date") < F.lit("2025-10-01"))
test_spark = df.filter(F.col("pickup_date") >= F.lit("2025-10-01"))

# Samples are independent of the source data; no source files are overwritten.
train_spark = train_spark.sample(False, 0.10, seed=42)
test_spark = test_spark.sample(False, 0.20, seed=43)

cols = features + ["trips", "total_revenue"]

print("Collecting training sample...")
train = train_spark.select(*cols).toPandas()

print("Collecting test sample...")
test = test_spark.select(*cols).toPandas()

print("Training sample rows:", len(train))
print("Test sample rows:", len(test))

if len(train) < 1000 or len(test) < 1000:
    raise RuntimeError("Sample too small; stop and inspect the logs.")

X_train = train[features].astype("float32")
X_test = test[features].astype("float32")

results = {}

for target in ["trips", "total_revenue"]:
    print(f"\nTraining XGBoost for {target}...")

    # Log target reduces the influence of very large values during training.
    y_train = np.log1p(train[target].clip(lower=0).to_numpy(dtype="float64"))
    y_test = test[target].clip(lower=0).to_numpy(dtype="float64")

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
        eval_metric="rmse"
    )

    model.fit(X_train, y_train, verbose=False)

    pred = np.maximum(0, np.expm1(model.predict(X_test)))

    mae = float(np.mean(np.abs(y_test - pred)))
    rmse = float(np.sqrt(np.mean((y_test - pred) ** 2)))

    denominator = float(np.sum((y_test - np.mean(y_test)) ** 2))
    r2 = (
        float(1 - np.sum((y_test - pred) ** 2) / denominator)
        if denominator > 0 else None
    )

    model_path = os.path.join(OUT, f"{target}_xgboost.json")
    model.save_model(model_path)

    results[target] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "train_sample_rows": len(train),
        "test_sample_rows": len(test),
        "features": features
    }

    print(f"{target} MAE:  {mae:.4f}")
    print(f"{target} RMSE: {rmse:.4f}")
    print(f"{target} R2:   {r2}")

with open(os.path.join(OUT, "metrics.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\nRESULTS")
print(json.dumps(results, indent=2))
print("\nModels saved under:", OUT)

spark.stop()
