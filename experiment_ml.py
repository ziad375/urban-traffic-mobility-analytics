from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

spark = SparkSession.builder.appName("NYCMLExperiments").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

source = "hdfs://namenode:8020/data/nyc/processed/zone_hour"

base_cols = [
    "trips", "avg_fare", "avg_distance",
    "avg_duration_min", "avg_speed_mph", "total_revenue"
]

df = spark.read.parquet(source)

for c in base_cols:
    df = df.withColumn(c, F.col(c).cast("double"))

df = df.dropna(subset=base_cols)

# Use one reproducible sample for a quick comparison.
sample = df.sample(withReplacement=False, fraction=0.02, seed=42).cache()
print("Experiment sample rows:", sample.count())

experiments = [
    ("baseline", base_cols),
    ("log_trips_revenue", base_cols),
    ("without_revenue", [
        "trips", "avg_fare", "avg_distance",
        "avg_duration_min", "avg_speed_mph"
    ])
]

evaluator = ClusteringEvaluator(
    featuresCol="features",
    predictionCol="prediction",
    metricName="silhouette",
    distanceMeasure="squaredEuclidean"
)

for name, cols in experiments:
    print(f"\n===== EXPERIMENT: {name} =====")

    work = sample
    if name == "log_trips_revenue":
        work = (
            sample
            .withColumn("trips", F.log1p(F.col("trips")))
            .withColumn(
                "total_revenue",
                F.log1p(F.greatest(F.col("total_revenue"), F.lit(0.0)))
            )
        )

    assembler = VectorAssembler(
        inputCols=cols,
        outputCol="raw_features",
        handleInvalid="skip"
    )
    assembled = assembler.transform(work)

    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withMean=True,
        withStd=True
    )
    scaler_model = scaler.fit(assembled)
    scaled = scaler_model.transform(assembled).cache()

    model = KMeans(
        k=5,
        seed=42,
        featuresCol="features",
        predictionCol="prediction",
        maxIter=20
    ).fit(scaled)

    predictions = model.transform(scaled).cache()
    score = evaluator.evaluate(predictions)

    print("Silhouette:", score)
    print("Cluster sizes:")
    predictions.groupBy("prediction").count().orderBy("prediction").show()

    predictions.unpersist()
    scaled.unpersist()

sample.unpersist()
spark.stop()
