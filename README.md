## a. Problem Statement

This project focuses on building a **binary classification model** that predicts whether a person earns more than $50,000 per year using census data.

The aim is to study demographic and work-related characteristics and use them to group individuals into two income levels:

* **≤ $50K** – People earning $50,000 or less annually
* **> $50K** – People earning more than $50,000 annually

Such a prediction system is useful in several real-world scenarios, including:

* Supporting economic studies and government policy decisions
* Enabling targeted marketing and better customer segmentation
* Gaining insights into the social and economic factors that influence income

To achieve this, the project trains and evaluates **six different machine learning models** and compares their performance to determine which approach works best for predicting income levels.

---

## b. Dataset Description

**Dataset Name**: Adult Census Income Dataset (UCI Machine Learning Repository)

**Source**:
[https://archive.ics.uci.edu/ml/datasets/adult](https://archive.ics.uci.edu/ml/datasets/adult)

### Dataset Characteristics

* **Total Instances**: 48,842 records
* **Total Features**: 14 features (before feature engineering)
* **Target Variable**: `income` (binary: <=50K, >50K)

### Class Distribution

* <=50K: 37,155 instances (76.1%)
* > 50K: 11,687 instances (23.9%)

### Missing Values

* Present in `workclass`, `occupation`, and `native-country` (represented as `?`)

### Data Split

* 85% training (41,515 samples)
* 15% testing (7,327 samples)
* Stratified split

---

### Features (Original Dataset)

| Feature         | Type        | Description               |
| --------------- | ----------- | ------------------------- |
| age             | Continuous  | Age in years              |
| workclass       | Categorical | Employment type           |
| fnlwgt          | Continuous  | Census sampling weight    |
| education       | Categorical | Highest education level   |
| educational-num | Continuous  | Education in numeric form |
| marital-status  | Categorical | Marital status            |
| occupation      | Categorical | Type of occupation        |
| relationship    | Categorical | Family relationship       |
| race            | Categorical | Race                      |
| gender          | Categorical | Male or Female            |
| capital-gain    | Continuous  | Capital gains             |
| capital-loss    | Continuous  | Capital losses            |
| hours-per-week  | Continuous  | Working hours per week    |
| native-country  | Categorical | Country of origin         |
   
---

### Feature Engineering Applied    

* Dropped redundant features: `education` (kept `educational-num`), `fnlwgt`
* Handled missing values: replaced `?` with `"Unknown"`
* Collapsed rare categories in `native-country` (threshold < 100 occurrences)
* Binary flags: `has_capital_gain`, `has_capital_loss`
* Log transformations: `log_capital_gain`, `log_capital_loss`
* Age buckets: 17–25, 26–35, 36–45, 46–55, 56–65, 65+
* Hours-per-week categories: part_time, full_time, overtime
* Interaction features: age × educational-num, hours × capital-gain, etc.
* Ratio features: capital_gain / capital_loss ratio

**Final Feature Count**: 51 features after engineering

---

## c. Models Used

* Logistic Regression
* Decision Tree
* k-Nearest Neighbor (kNN)
* Naive Bayes
* Random Forest (Ensemble)
* XGBoost (Ensemble)

---

### Comparison Table with Evaluation Metrics

| ML Model                 | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| ------------------------ | -------- | ------ | --------- | ------ | ------ | ------ |
| Logistic Regression      | 0.7824   | 0.8497 | 0.5341    | 0.7100 | 0.6096 | 0.4719 |
| Decision Tree            | 0.8119   | 0.7400 | 0.6079    | 0.6022 | 0.6051 | 0.4816 |
| kNN                      | 0.7982   | 0.7705 | 0.6206    | 0.4029 | 0.4886 | 0.3833 |
| Naive Bayes              | 0.7932   | 0.8307 | 0.6427    | 0.3062 | 0.4148 | 0.3392 |
| Random Forest (Ensemble) | 0.8637   | 0.9197 | 0.7928    | 0.5825 | 0.6716 | 0.5992 |
| XGBoost (Ensemble)       | 0.8680   | 0.9273 | 0.7781    | 0.6270 | 0.6945 | 0.6173 |

---

### Best Models by Metric

* **Accuracy**: XGBoost (86.80%)
* **AUC**: XGBoost (92.73%)
* **Precision**: Random Forest (79.28%)
* **Recall**: Logistic Regression (71.00%)
* **F1 Score**: XGBoost (69.45%)
* **MCC**: XGBoost (61.73%)


---

### Model Performance Observations

| ML Model Name            | Observation about model performance |
| ------------------------ | ----------------------------------- |
| Logistic Regression      | Achieves good recall (0.7100) but lower precision (0.5341), indicating it captures most positive cases but with more false positives. Best for scenarios prioritizing sensitivity. |
| Decision Tree            | Shows balanced precision (0.6079) and recall (0.6022) with moderate accuracy (0.8119). However, lower AUC (0.7400) suggests potential overfitting to training data. |
| kNN                      | Struggles with low recall (0.4029) despite decent precision (0.6206), missing many positive cases. Poor F1 score (0.4886) indicates suboptimal performance for imbalanced data. |
| Naive Bayes              | Exhibits highest precision (0.6427) among single models but severely limited recall (0.3062). Makes conservative predictions, missing majority of positive income cases. |
| Random Forest (Ensemble) | Strong overall performer with excellent AUC (0.9197) and balanced metrics. High precision (0.7928) makes it reliable for positive predictions with good generalization capability. |
| XGBoost (Ensemble)       | Top performer across all metrics with highest accuracy (0.8680) and AUC (0.9273). Best balance of precision-recall trade-off, making it the most robust model. |

---