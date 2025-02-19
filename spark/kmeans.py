from pyspark.ml.clustering import KMeans
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
import numpy as np
from pyspark.ml.evaluation import ClusteringEvaluator
import matplotlib.pyplot as plt
from pandas.plotting import table
import time
from fpdf import FPDF

# Use 5 executors (as integer, not string)
executors = 5

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("KMeansExample") \
    .master(f"local[{executors}]").getOrCreate()

# Load dataset (adjust path to your dataset)
dataset = spark.read.csv("./data/Electric_Vehicle_Population_Data.csv", header=True, inferSchema=True)

# Remove rows where 'Electric Range' or 'Model Year' is null or 0
dataset = dataset.filter((dataset['Electric Range'].isNotNull()) & (dataset['Model Year'].isNotNull()))

# Ensure the columns are of numeric types
dataset = dataset.withColumn("Model Year", dataset["Model Year"].cast("double"))
dataset = dataset.withColumn("Electric Range", dataset["Electric Range"].cast("double"))

# Repartition dataset for distributed execution
dataset = dataset.repartition(5)
assembler = VectorAssembler(inputCols=['Model Year', 'Electric Range'], outputCol='features')
dataset = assembler.transform(dataset)

start_time = time.time()

# Run KMeans clustering
kmeans = KMeans().setK(4).setSeed(1)
model = kmeans.fit(dataset)
predictions = model.transform(dataset)

# Show the results
predictions.select("features", "prediction").show(predictions.count(), truncate=False)
centers = model.clusterCenters()
for center in centers:
    print("Cluster Center: ", center)

end_time = time.time()
# Convert Spark DataFrame to Pandas for easier plotting
predictions_pd = predictions.select("Model Year", "Electric Range", "prediction").toPandas()


# Create a ClusteringEvaluator to calculate the Silhouette Score
evaluator = ClusteringEvaluator()

# Compute Silhouette Score
silhouette = evaluator.evaluate(predictions)
print(f"Silhouette Score: {silhouette}")

execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

# # Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(predictions_pd['Model Year'], predictions_pd['Electric Range'], c=predictions_pd['prediction'], cmap='viridis')
plt.xlabel('Model Year')
plt.ylabel('Electric Range')
plt.title('KMeans Clustering of Electric Vehicles')
plt.colorbar(label='Cluster')
plt.show()


spark.stop()
