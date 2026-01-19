# kaggle-heart-attack-ml-byme-


# Heart Attack Prediction using Logistic Regression (Machine Learning)

## 📌 Project Overview

This project focuses on building a **classification model** using **Logistic Regression** to predict the likelihood of a **heart attack** based on medical data. The dataset used for this project was sourced from **Kaggle**.

The goal of this project is to understand the complete **supervised machine learning classification workflow**, including data preprocessing, model training, prediction, and evaluation using appropriate classification metrics.

---

## 🛠️ Tools & Libraries Used

* **Python**
* **Pandas** – data loading and manipulation
* **Seaborn** – data visualization
* **Scikit-learn** – model building, training, and evaluation
* **JupyterLab** – development environment

```python
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
```

---

## 📂 Dataset

* Dataset: **Heart Attack Dataset**
* Source: **Kaggle**
* Target variable: Presence or absence of heart attack (binary classification)

---

## 🔄 Project Workflow

### 1️⃣ Data Loading & Exploration

* Loaded dataset using **Pandas**
* Checked data structure, data types, and missing values
* Performed basic data visualization using **Seaborn**

---

### 2️⃣ Feature & Target Separation

* Separated input features (**X**) and target variable (**y**)

```python
X = data.drop(columns=["target"])
y = data["target"]
```

---

### 3️⃣ Train-Test Split

* Split the dataset into training and testing sets

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

### 4️⃣ Model Training – Logistic Regression

* Trained the classification model using **Logistic Regression**

```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

---

### 5️⃣ Prediction

* Predicted outcomes on the test dataset

```python
y_pred = model.predict(X_test)
```

---

### 6️⃣ Model Evaluation

#### 📊 Accuracy Score

* Measures overall correctness of predictions

```python
accuracy = accuracy_score(y_test, y_pred)
```

#### 📊 Precision Score

* Measures how many predicted positive cases are actually positive

```python
precision = precision_score(y_test, y_pred)
```

---

## ✅ Results & Insights

* Logistic Regression performed well for binary classification
* Accuracy provided overall model performance
* Precision helped evaluate prediction quality for positive (heart attack) cases

---

## 🧠 Key Learnings

* Difference between **regression vs classification** problems
* Importance of **train-test split**
* Understanding **accuracy vs precision** metrics
* Applying Logistic Regression to real-world healthcare data

---

## 🚀 Conclusion

This project strengthened my understanding of **classification algorithms** and evaluation metrics. Logistic Regression proved to be an effective baseline model for predicting heart attack risk using structured medical data.

---

## 📌 Future Improvements

* Add **Recall and F1-score**
* Perform **feature scaling**
* Use **confusion matrix** for deeper evaluation
* Compare with other classifiers (Decision Tree, Random Forest)

---

⭐ If you find this project useful, feel free to star the repository!
