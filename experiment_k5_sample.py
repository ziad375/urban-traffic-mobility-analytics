from pyspark.sql import SparkSession, functions as F
from pyspark import StorageLevel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

spark = SparkSession.builder.appName("NYCTrafficK5Validation").getOrCreate()
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

sample = df.sample(False, 0.05, 42).persist(
    StorageLevel.DISK_ONLY
)
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

scaled = scaler.fit(assembled).transform(assembled).persist(
    StorageLevel.DISK_ONLY
)
print("Evaluation rows:", scaled.count())

model = KMeans(
    k=5,
    seed=42,
    featuresCol="features",
    predictionCol="prediction",
    maxIter=20
).fit(scaled)

predictions = model.transform(scaled)

evaluator = ClusteringEvaluator(
    featuresCol="features",
    predictionCol="prediction",
    metricName="silhouette",
    distanceMeasure="squaredEuclidean"
)

print("K=5 Silhouette:", evaluator.evaluate(predictions))
print("Cluster sizes:")
predictions.groupBy("prediction").count().orderBy("prediction").show()

scaled.unpersist()
sample.unpersist()
spark.stop()
