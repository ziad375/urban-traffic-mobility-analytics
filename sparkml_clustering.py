from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

spark = SparkSession.builder \
    .appName("NYCTaxiKMeansClustering") \
    .getOrCreate()

input_path = "hdfs://namenode:8020/data/nyc/mahout/features"
output_path = "hdfs://namenode:8020/data/nyc/sparkml/clusters"

# Read the standardized features prepared earlier
df = spark.read.parquet(input_path).dropna(subset=["features"])

# Train K-Means
kmeans = KMeans(
    k=5,
    seed=42,
    featuresCol="features",
    predictionCol="prediction"
)

model = kmeans.fit(df)
predictions = model.transform(df)

# Evaluate clustering quality
evaluator = ClusteringEvaluator(
    featuresCol="features",
    predictionCol="prediction",
    metricName="silhouette"
)

silhouette = evaluator.evaluate(predictions)

# Save predictions to HDFS
predictions.select(
    "pu_zone",
    "pickup_date",
    "pickup_hour",
    "prediction"
).write.mode("overwrite").parquet(output_path)

print("Number of records:", predictions.count())
print("Number of clusters:", kmeans.getK())
print("Silhouette score:", silhouette)
print("Cluster centers:")
for center in model.clusterCenters():
    print(center)

print("Predictions saved to:", output_path)

spark.stop()