from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, StringIndexer
from pyspark.sql.functions import split, concat_ws, udf
from pyspark.ml.linalg import Vector
from pyspark.sql.types import StringType

# -----------------------------------
# Step 1: Initialize Spark session
# -----------------------------------
spark = SparkSession.builder.appName("FakeNews_Task3").getOrCreate()

# -----------------------------------
# Step 2: Load the CSV output from Task 2
# -----------------------------------
df = spark.read.option("header", True).csv("task2_output.csv")

# -----------------------------------
# Step 3: Convert filtered_words_str back to an array of words
# -----------------------------------
df = df.withColumn("filtered_words", split(df["filtered_words_str"], " "))

# -----------------------------------
# Step 4: Apply HashingTF
# -----------------------------------
hashing_tf = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
df_featurized = hashing_tf.transform(df)

# -----------------------------------
# Step 5: Apply IDF
# -----------------------------------
idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(df_featurized)
df_tfidf = idf_model.transform(df_featurized)

# -----------------------------------
# Step 6: Index labels (FAKE -> 0.0, REAL -> 1.0)
# -----------------------------------
indexer = StringIndexer(inputCol="label", outputCol="label_index")
df_final = indexer.fit(df_tfidf).transform(df_tfidf)

# -----------------------------------
# Step 7: Convert the sparse vector to string before saving
# -----------------------------------

# UDF to convert vector to string
def vector_to_string(v):
    return str(v)

vector_to_string_udf = udf(vector_to_string, StringType())

# Create final DataFrame with string versions of arrays/vectors
df_output = df_final.withColumn(
    "filtered_words_str_final",
    concat_ws(" ", "filtered_words")
).withColumn(
    "features_str",
    vector_to_string_udf("features")
)

# -----------------------------------
# Step 8: Save to CSV
# -----------------------------------
df_output.select("id", "filtered_words_str_final", "features_str", "label_index") \
    .write.option("header", True).mode("overwrite").csv("task3_output.csv")

print("Task 3 output saved to task3_output.csv")
