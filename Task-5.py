from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# -----------------------------------
# Step 1: Initialize Spark session
# -----------------------------------
spark = SparkSession.builder.appName("FakeNews_Task5").getOrCreate()

# -----------------------------------
# Step 2: Load predictions from Task 4
# -----------------------------------
df = spark.read.option("header", True).csv("task4_output.csv")

# -----------------------------------
# Step 3: Convert label_index and prediction to DoubleType
# -----------------------------------
df = df.withColumn("label_index", df["label_index"].cast("double"))
df = df.withColumn("prediction", df["prediction"].cast("double"))

# -----------------------------------
# Step 4: Evaluate Accuracy
# -----------------------------------
accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol="label_index", predictionCol="prediction", metricName="accuracy"
)
accuracy = accuracy_evaluator.evaluate(df)

# -----------------------------------
# Step 5: Evaluate F1 Score
# -----------------------------------
f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label_index", predictionCol="prediction", metricName="f1"
)
f1_score = f1_evaluator.evaluate(df)

# -----------------------------------
# Step 6: Print results
# -----------------------------------
print("Model Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1_score:.4f}")

# -----------------------------------
# Step 7: Save metrics to CSV
# -----------------------------------
metrics = spark.createDataFrame(
    [("Accuracy", round(accuracy, 4)), ("F1 Score", round(f1_score, 4))],
    ["Metric", "Value"]
)

metrics.write.option("header", True).mode("overwrite").csv("task5_output.csv")

print("Task 5 evaluation metrics saved to task5_output.csv")
