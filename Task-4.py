from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vectors, SparseVector, VectorUDT
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
import re

# -----------------------------------
# Step 1: Initialize Spark session
# -----------------------------------
spark = SparkSession.builder.appName("FakeNews_Task4").getOrCreate()

# -----------------------------------
# Step 2: Load the CSV output from Task 3
# -----------------------------------
df = spark.read.option("header", True).csv("task3_output.csv")

# -----------------------------------
# Step 3: Convert features_str back to SparseVector
# -----------------------------------

def parse_sparse_vector(s):
    """
    Convert string like (10000,[3,47,88],[0.5,1.2,0.8]) into a SparseVector.
    """
    if s is None:
        return None
    pattern = r"\((\d+),\[(.*?)\],\[(.*?)\]\)"
    match = re.match(pattern, s)
    if not match:
        return None
    size = int(match.group(1))
    indices = [int(i) for i in match.group(2).split(",")] if match.group(2) else []
    values = [float(v) for v in match.group(3).split(",")] if match.group(3) else []
    return SparseVector(size, indices, values)

# VERY IMPORTANT: Specify return type as VectorUDT
parse_sparse_vector_udf = udf(parse_sparse_vector, VectorUDT())

# Apply UDF to convert back to SparseVector
df = df.withColumn("features", parse_sparse_vector_udf("features_str"))

# -----------------------------------
# Step 4: Convert label_index to DoubleType
# -----------------------------------
df = df.withColumn("label_index", df["label_index"].cast(DoubleType()))

# -----------------------------------
# Step 5 (Debug): Check the schema to confirm 'features' is a vector
# -----------------------------------
df.printSchema()

# -----------------------------------
# Step 6: Train/test split
# -----------------------------------
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# -----------------------------------
# Step 7: Train Logistic Regression
# -----------------------------------
lr = LogisticRegression(featuresCol="features", labelCol="label_index")
model = lr.fit(train_data)

# -----------------------------------
# Step 8: Predict on test data
# -----------------------------------
predictions = model.transform(test_data)

# -----------------------------------
# Step 9: Save predictions to CSV
# -----------------------------------
# We need the original title (from Task 2), so reload it:
df_task2 = spark.read.option("header", True).csv("task2_output.csv")

# Join predictions with titles (matching by id)
predictions = predictions.join(df_task2.select("id", "title"), on="id", how="left")

predictions.select("id", "title", "label_index", "prediction") \
    .write.option("header", True).mode("overwrite").csv("task4_output.csv")

print(" Task 4 predictions saved to task4_output.csv")
