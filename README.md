# Assignment-5-FakeNews-Detection
---
## **Prerequisites**

Before starting the assignment, ensure the following tools are installed:

1. **Python 3.x**

   * [Download Python](https://www.python.org/downloads/)
   * Verify installation:

     ```bash
     python --version
     ```

2. **PySpark**

   * Install using pip:

     ```bash
     pip install pyspark
     ```

3. **(Optional) pandas** (for viewing CSV outputs locally)

   * Install using pip:

     ```bash
     pip install pandas
     ```

---

## **Setup Instructions**

* **Task-1.py**: Load and explore the dataset.
* **Task-2.py**: Clean and tokenize the news text.
* **Task-3.py**: Extract TF-IDF features and prepare data for modeling.
* **Task-4.py**: Train a Logistic Regression model and predict.
* **Task-5.py**: Evaluate the model (Accuracy and F1 Score).
* **Output CSVs**: task1\_output.csv → task5\_output.csv

---

### **2. Running the Tasks**

Run each script in sequence:

```bash
python Task-1.py
python Task-2.py
python Task-3.py
python Task-4.py
python Task-5.py
```

---

## **Overview**

In this assignment, we build a **Spark MLlib pipeline** for classifying news articles as **FAKE** or **REAL** based on their content. The system ingests text data, preprocesses it, extracts features using TF-IDF, trains a Logistic Regression model, and evaluates its performance.

This assignment builds skills in large-scale text processing, feature engineering, and binary classification using PySpark.

---

## **Objectives**

1. Load and explore news data.
2. Clean and tokenize the article text.
3. Extract TF-IDF features and encode labels.
4. Train a binary classifier (Logistic Regression).
5. Evaluate the model using Accuracy and F1 Score.

---

## **Dataset Structure**

| Field | Type   | Description                   |
| ----- | ------ | ----------------------------- |
| id    | String | Unique article ID             |
| title | String | Headline of the article       |
| text  | String | Full text of the news article |
| label | String | Label (`FAKE` or `REAL`)      |

---

## **Assignment Tasks**

---

### **1. Load & Basic Exploration**

**Objective:**

* Read the CSV file (`fake_news_sample.csv`).
* Display sample records.
* Count total articles and unique labels.

**Techniques Used:**

* `spark.read.csv()`
* `createOrReplaceTempView()`
* `show()` and `count()`

**Output CSV:** `task1_output.csv`

![image](https://github.com/user-attachments/assets/21d5fa3a-d742-4921-9f39-5755bf9485f5)


---

### **2. Text Preprocessing**

**Objective:**

* Combine title and text.
* Convert to lowercase.
* Tokenize text.
* Remove stopwords.

**Techniques Used:**

* `Tokenizer`
* `StopWordsRemover`
* `concat_ws()`, `lower()`

**Output CSV:** `task2_output.csv`

![image](https://github.com/user-attachments/assets/5d73e698-da09-4cb6-9d75-1ef31535479e)


---

### **3. Feature Extraction**

**Objective:**

* Apply **HashingTF** for term frequencies.
* Apply **IDF** for scaling.
* Convert labels to numeric using **StringIndexer**.

**Techniques Used:**

* `HashingTF`
* `IDF`
* `StringIndexer`

**Output CSV:** `task3_output.csv`

![image](https://github.com/user-attachments/assets/dd3a6392-31f4-4dbd-9eb1-b2b297f749b4)


---

### **4. Model Training**

**Objective:**

* Split data into training and testing sets.
* Train **Logistic Regression**.
* Generate predictions.

**Techniques Used:**

* `randomSplit()`
* `LogisticRegression`

**Output CSV:** `task4_output.csv`

![image](https://github.com/user-attachments/assets/433fe88e-fa5a-4446-bbc3-f943f935d1f9)


---

### **5. Model Evaluation**

**Objective:**

* Compute Accuracy and F1 Score.
* Save metrics.

**Techniques Used:**

* `MulticlassClassificationEvaluator`

**Output CSV:** `task5_output.csv`

**Sample Output:**

| Metric   | Value |
| -------- | ----- |
| Accuracy | 0.89  |
| F1 Score | 0.88  |

![image](https://github.com/user-attachments/assets/50440e8c-433e-4014-b880-873721ace84a)

![image](https://github.com/user-attachments/assets/8402e764-c38c-4385-b6b6-229b64c659e3)

---

## **Conclusion**

This assignment provided hands-on experience in building a complete **text classification pipeline** using PySpark. The process involved data ingestion, text cleaning, feature engineering, model training, prediction, and evaluation.

It strengthened skills in:

* Text preprocessing at scale
* Feature extraction with TF-IDF
* Model training and evaluation
* Managing data types and saving structured outputs

---
