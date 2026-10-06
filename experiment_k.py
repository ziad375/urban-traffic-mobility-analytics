from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

spark = SparkSession.builder.appName("NYCKSelection").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

df = spark.read.parquet(
    "hdfs://namenode:8020/data/nyc/processed/zone_hour"
)

cols = [
    "trips", "avg_fare", "avg_distance",
    "avg_duration_min", "avg_speed_mph", "total_revenue"
]

for c in cols:
    df = df.withColumn(c, F.col(c).cast("double"))

df = df.dropna(subset=cols)

sample = df.sample(False, 0.02, 42).cache()
print("Sample rows:", sample.count())

assembler = VectorAssembler(
    inputCols=cols,
    outputCol="raw_features",
    handleInvalid="skip"
)
assembled = assembler.transform(sample)

scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withMean=True,
    withStd=True
)
scaled = scaler.fit(assembled).transform(assembled).cache()

evaluator = ClusteringEvaluator(
    featuresCol="features",
    predictionCol="prediction",
    metricName="silhouette",
    distanceMeasure="squaredEuclidean"
)

for k in [3, 4, 5, 6, 7, 8]:
    model = KMeans(
        k=k,
        seed=42,
        featuresCol="features",
        predictionCol="prediction",
        maxIter=20
    ).fit(scaled)

    predictions = model.transform(scaled).cache()
    score = evaluator.evaluate(predictions)

    print(f"K={k}, Silhouette={score:.6f}")
    print(f"K={k} cluster sizes:")
    predictions.groupBy("prediction").count().orderBy("prediction").show()

    predictions.unpersist()

scaled.unpersist()
sample.unpersist()
spark.stop()
