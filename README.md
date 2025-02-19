# KMeans Clustering on Electric Vehicle Population Data

This project demonstrates the use of KMeans clustering on the Electric Vehicle Population Data using PySpark. The goal is to cluster vehicles based on their `Model Year` and `Electric Range`.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Code Overview](#code-overview)
- [Results](#results)
- [License](#license)

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
