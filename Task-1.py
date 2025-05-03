from pyspark.sql import SparkSession

# Step 1: Initialize Spark
spark = SparkSession.builder.appName("FakeNewsClassification").getOrCreate()

# Step 2: Load the CSV
df = spark.read.option("header", True).option("inferSchema", True).csv("fake_news_sample.csv")

# Step 3: Create Temporary View
df.createOrReplaceTempView("news_data")

# Step 4: Basic Queries
df.show(5)  # Show first 5 rows
print(f"Total articles: {df.count()}")  # Count rows

spark.sql("SELECT DISTINCT label FROM news_data").show()  # Distinct labels

# Step 5: Save output
df.limit(5).write.option("header", True).csv("task1_output.csv")
