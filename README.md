# 🍷 Wine Quality Prediction (Random Forest + Hyperparameter Tuning)

## 📌 Project Overview

This project builds a Machine Learning model to predict **wine quality** using physicochemical properties from the WineQT dataset.

The pipeline includes:

* Exploratory Data Analysis (EDA)
* Outlier Detection & Handling (Z-Score + IQR)
* Decision Tree & Random Forest Modeling
* Hyperparameter Optimization using RandomizedSearchCV
* Feature Importance Analysis
* Model Saving using Joblib

---

## 📂 Dataset

* Dataset: `WineQT.csv`
* Target variable: `quality`
* Features include:

  * Fixed acidity
  * Volatile acidity
  * Citric acid
  * Residual sugar
  * Chlorides
  * Free sulfur dioxide
  * Total sulfur dioxide
  * Density
  * pH
  * Sulphates
  * Alcohol

---

# 🔍 1️⃣ Exploratory Data Analysis (EDA)

### ✔ Dataset Inspection

* `.head()`
* `.info()`
* `.describe()`
* Missing value check

### ✔ Correlation Heatmap

Used to understand relationships between variables.

```python
sns.heatmap(data=df.corr(), annot=True)
```

### ✔ Outlier Detection

* Boxplot visualization
* Skewness analysis

### ✔ Distribution Check

* Histogram + KDE
* Skewness printed for each column

---

# ⚙️ 2️⃣ Outlier Handling Strategy

We used a **conditional approach**:

| Data Distribution | Method Used |        |                          |
| ----------------- | ----------- | ------ | ------------------------ |
| Nearly Normal (   | skew        | ≤ 0.3) | Z-score Capping (±3 std) |
| Skewed Data       | IQR Method  |        |                          |

### ✔ Z-score Capping

[
Z = \frac{X - \mu}{\sigma}
]

Values beyond ±3 standard deviations were capped.

### ✔ IQR Method

[
IQR = Q3 - Q1
]
Bounds:

* Lower = Q1 − 1.5 × IQR
* Upper = Q3 + 1.5 × IQR

---

# 🤖 3️⃣ Model Building

## 🎯 Problem Type

Converted multi-class wine quality into binary classification:

* 0 → Quality ≤ 5
* 1 → Quality > 5

---

## 🌳 Decision Tree Classifier

```python
DecisionTreeClassifier(class_weight='balanced')
```

Evaluation Metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* Classification Report

---

## 🌲 Random Forest Classifier

Initial model trained with default parameters.

```python
RandomForestClassifier()
```

---

# 🚀 4️⃣ Hyperparameter Tuning (RandomizedSearchCV)

Used Randomized Search with 5-fold Cross Validation.

### Parameters Tuned:

* `n_estimators`
* `max_depth`
* `max_features`
* `min_samples_split`
* `min_samples_leaf`
* `bootstrap`

```python
RandomizedSearchCV(
    n_iter=50,
    cv=5,
    n_jobs=-1,
    random_state=42
)
```

### 🎯 Why RandomizedSearch?

* Faster than GridSearch
* Explores large parameter space efficiently
* Better generalization via cross-validation

---

# 📊 5️⃣ Model Evaluation

Metrics used:

* Accuracy
* Precision (Weighted)
* Recall (Weighted)
* F1 Score
* Classification Report

---

# 📈 6️⃣ Feature Importance

Random Forest provides feature importance scores.

```python
best_rf_model.feature_importances_
```

Top important features driving wine quality were visualized using a bar plot.

---

# 💾 7️⃣ Model Saving

The best optimized model was saved using Joblib:

```python
joblib.dump(best_rf_model, 'wine_quality_rf_76acc.pkl')
```

Saved file:

```
wine_quality_rf_76acc.pkl
```

---

# 🧠 Key Learnings

* Outlier handling improves model stability
* Class imbalance must be handled (`class_weight='balanced'`)
* Hyperparameter tuning significantly improves performance
* Random Forest generally outperforms single Decision Trees
* Feature importance helps interpret black-box models

---

# 🛠 Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* SciPy
* Joblib

---

# 🔮 Future Improvements

* Try XGBoost / LightGBM
* Use SMOTE for imbalance handling
* Apply Feature Scaling + PCA
* Deploy model using Streamlit
* Convert to multiclass prediction instead of binary

---

# 👨‍💻 Author

**Sarthak Shukla**
B.Tech CSDS | Machine Learning Enthusiast

---

---
