# KMeans Clustering on Electric Vehicle Population Data

This project demonstrates the use of KMeans clustering on the Electric Vehicle Population Data using PySpark. The goal is to cluster vehicles based on their `Model Year` and `Electric Range`.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Code Overview](#code-overview)
- [Results](#results)

## Installation

To run this project, you need to have the following installed:

1. **Python**: Ensure you have Python 3.7 or higher installed.
2. **PySpark**: Install PySpark using pip:
   ```bash
   pip install pyspark
   ```
Additional Libraries: Install the required libraries:
```bash
   pip install numpy matplotlib
```
## Usage
 1. Clone the Repository:
```bash
   git clone https://github.com/B3krodge/distributedKmeans.git
   cd distributedKmeans
```
 2. Download the Dataset:
Download the Electric_Vehicle_Population_Data.csv dataset and place it in the ./data/ directory. You can find this dataset here: https://www.kaggle.com/datasets/ratikkakkar/electric-vehicle-population-data
 3. Run the Script:
```bash
python kmeans_clustering.py
```
## Code Overview
The script performs the following steps:

Initialize SparkSession:
``` bash
spark = SparkSession.builder \
    .appName("KMeansExample") \
    .master(f"local[{executors}]").getOrCreate()
```
Load and Preprocess Data:

1. Load the dataset and filter out rows with null or zero values in Electric Range or Model Year.

2. Cast the relevant columns to numeric types.

3. Repartition the dataset for distributed execution.

Feature Assembly:
```bash
assembler = VectorAssembler(inputCols=['Model Year', 'Electric Range'], outputCol='features')
dataset = assembler.transform(dataset)
```
KMeans Clustering:

1. Fit the KMeans model with 4 clusters.

2. Transform the dataset to include cluster predictions.

Evaluate Clustering:

1. Calculate the Silhouette Score to evaluate the clustering performance.

Visualize Results:

1. Plot the clusters using matplotlib.

```bash 
spark.stop()
```
## Results
1. Cluster Centers: The script prints the center of each cluster.

2. Silhouette Score: The Silhouette Score is printed to evaluate the clustering quality.

3. Execution Time: The total execution time is printed.

4. Visualization: A scatter plot is displayed showing the clusters.
