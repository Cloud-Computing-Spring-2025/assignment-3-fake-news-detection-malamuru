from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.sql.functions import lower, concat_ws

# -----------------------------------
# Step 1: Initialize Spark session
# -----------------------------------
spark = SparkSession.builder.appName("FakeNews_Task2").getOrCreate()

# -----------------------------------
# Step 2: Load the CSV again
# -----------------------------------
df = spark.read.option("header", True).option("inferSchema", True).csv("fake_news_sample.csv")

# -----------------------------------
# Step 3: Combine title + text into a single lowercase column
# -----------------------------------
df = df.withColumn("full_text", lower(concat_ws(" ", "title", "text")))

# -----------------------------------
# Step 4: Tokenize the text into words
# -----------------------------------
tokenizer = Tokenizer(inputCol="full_text", outputCol="words")
df_words = tokenizer.transform(df)

# -----------------------------------
# Step 5: Remove stopwords
# -----------------------------------
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
df_clean = remover.transform(df_words)

# -----------------------------------
# Step 6: Convert array column to string before saving to CSV
# -----------------------------------
from pyspark.sql.functions import concat_ws

df_output = df_clean.withColumn(
    "filtered_words_str",
    concat_ws(" ", "filtered_words")  # Join list of words into a string
)

# -----------------------------------
# Step 7: Save required output columns to CSV
# -----------------------------------
df_output.select("id", "title", "filtered_words_str", "label") \
    .write.option("header", True).mode("overwrite").csv("task2_output.csv")

print(" Task 2 output saved to task2_output.csv")
