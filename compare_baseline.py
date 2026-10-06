import numpy as np
import pandas as pd
import xgboost as xgb

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("CompareXGBoostBaseline").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

df = spark.read.parquet(
    "hdfs://namenode:8020/data/nyc/processed/zone_hour"
)

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
                F.lit(":00:00"
                )
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
    df.withColumn("prev_trips", F.lag("trips").over(w))
      .withColumn("prev_revenue", F.lag("total_revenue").over(w))
      .withColumn("prev_ts", F.lag("ts").over(w))
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
    "pu_zone", "pickup_hour", "day_of_week", "day_of_month",
    "month", "year", "date_number", "prev_trips",
    "prev_revenue", "hours_since_prev"
]

train = (
    df.filter(F.col("pickup_date") < "2025-10-01")
      .sample(False, 0.10, seed=42)
      .select("trips", "total_revenue")
      .toPandas()
)

test = (
    df.filter(F.col("pickup_date") >= "2025-10-01")
      .sample(False, 0.20, seed=43)
      .select(*(features + ["trips", "total_revenue"]))
      .toPandas()
)

print("Train sample:", len(train))
print("Test sample:", len(test))

X_test = test[features].astype("float32")

def metrics(actual, predicted):
    actual = np.asarray(actual, dtype="float64")
    predicted = np.asarray(predicted, dtype="float64")
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    denom = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - np.sum((actual - predicted) ** 2) / denom if denom else float("nan")
    return {"MAE": round(float(mae), 4),
            "RMSE": round(float(rmse), 4),
            "R2": round(float(r2), 6)}

for target, previous in [
    ("trips", "prev_trips"),
    ("total_revenue", "prev_revenue")
]:
    actual = test[target].to_numpy(dtype="float64")
    baseline_last = test[previous].to_numpy(dtype="float64")
    baseline_mean = np.full(len(test), train[target].mean())

    model = xgb.XGBRegressor()
    model.load_model(f"/tmp/xgboost_models/{target}_xgboost.json")
    prediction = np.maximum(0, np.expm1(model.predict(X_test)))

    print(f"\n===== {target} =====")
    print("Last available value:", metrics(actual, baseline_last))
    print("Training mean:", metrics(actual, baseline_mean))
    print("XGBoost:", metrics(actual, prediction))

spark.stop()
