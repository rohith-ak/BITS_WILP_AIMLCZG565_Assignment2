# Adult Census Income Classification

---

## Table of Contents

* [Problem Statement](#problem-statement)
* [Dataset Description](#dataset-description)
* [Models Used](#models-used)
* [Model Performance Observations](#model-performance-observations)
* [Installation & Setup](#installation--setup)
* [Usage](#usage)
* [Project Structure](#project-structure)
* [Deployment](#deployment)
* [Requirements](#requirements)
* [Features](#features)
* [License](#license)
* [Author](#author)
* [Acknowledgments](#acknowledgments)
* [Assignment Submission Checklist](#assignment-submission-checklist)

---

## Problem Statement

This project implements a **binary classification system** to predict whether an individual's annual income exceeds **$50,000** based on census data.

The goal is to analyze demographic and employment-related features to classify individuals into two income categories:

* **<=50K**: Annual income of $50,000 or less
* **>50K**: Annual income exceeding $50,000

This classification task is valuable for:

* Economic research and policy-making
* Targeted marketing and customer segmentation
* Understanding socioeconomic factors affecting income levels

The project implements and compares **6 different machine learning algorithms** to identify the most effective approach for income prediction.

---

## Dataset Description

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

## Features (Original Dataset)

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

## Feature Engineering Applied

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

## Models Used

* Logistic Regression
* Decision Tree
* k-Nearest Neighbor (kNN)
* Naive Bayes
* Random Forest (Ensemble)
* XGBoost (Ensemble)

---

## Comparison Table with Evaluation Metrics

| ML Model                 | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| ------------------------ | -------- | ------ | --------- | ------ | ------ | ------ |
| Logistic Regression      | 0.7763   | 0.8377 | 0.5248    | 0.6916 | 0.5968 | 0.4542 |
| Decision Tree            | 0.8185   | 0.7479 | 0.6229    | 0.6125 | 0.6176 | 0.4987 |
| kNN                      | 0.7912   | 0.7548 | 0.5958    | 0.3965 | 0.4761 | 0.3636 |
| Naive Bayes              | 0.7925   | 0.8249 | 0.6356    | 0.3118 | 0.4184 | 0.3387 |
| Random Forest (Ensemble) | 0.8689   | 0.9188 | 0.7905    | 0.6151 | 0.6918 | 0.6180 |
| XGBoost (Ensemble)       | 0.8768   | 0.9299 | 0.7962    | 0.6518 | 0.7168 | 0.6443 |

---

## Best Models by Metric

* **Accuracy**: XGBoost (87.68%)
* **AUC**: XGBoost (92.99%)
* **Precision**: XGBoost (79.62%)
* **Recall**: Logistic Regression (69.16%)
* **F1 Score**: XGBoost (71.68%)
* **MCC**: XGBoost (64.43%)

---

## Model Performance Observations

* Logistic Regression achieved moderate accuracy with strong recall.
* Decision Tree improved accuracy but showed weaker probability calibration.
* kNN suffered from poor recall and sensitivity to feature scaling.
* Naive Bayes underperformed due to independence assumptions.
* Random Forest delivered strong balanced performance.
* XGBoost emerged as the **best overall model**, excelling across most metrics.

---

## Installation & Setup

### Prerequisites

* Python 3.8 or higher
* pip package manager
* Virtual environment (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/adult-census-ml-classification.git
cd adult-census-ml-classification/project-folder
```

### Step 2: Create Virtual Environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Train All Models

```bash
cd model/src
python train.py
```

**Expected Output (excerpt)**:

```
TRAINING AND SAVING ALL MODELS
Loading training data...
Features shape: (41515, 51)
Target shape: (41515,)
Training: Logistic Regression
Training completed!
ALL MODELS TRAINED AND SAVED SUCCESSFULLY!
```

### Step 5: Generate Comparison Table (Optional)

```bash
python generate_comparison_table.py
```

Results saved to:

```
model/results/model_comparison.csv
```

---

## Usage

### Running the Streamlit App Locally

```bash
streamlit run app.py
```

App opens at:

```
http://localhost:8501
```

---

## Using the Application

1. Download test dataset (7,327 rows, 15 columns)
2. Upload CSV (same features as training data)
3. Select model:

   * Logistic Regression
   * Decision Tree
   * kNN
   * Naive Bayes
   * Random Forest
   * XGBoost
4. View model status
5. Predict & evaluate
6. View results:

   * Accuracy, Precision, Recall, F1, MCC, AUC
   * Confusion matrix (Plotly)
   * Classification report
7. Download predictions with probabilities

---

## Project Structure

```
project-folder/
│
├── app.py
├── requirements.txt
├── README.md
│
├── model/
│   ├── data/
│   │   ├── adult.csv
│   │   ├── adult_train.csv
│   │   └── adult_test.csv
│   │
│   ├── saved_models/
│   │   ├── logistic_regression.joblib
│   │   ├── decision_tree.joblib
│   │   ├── knn.joblib
│   │   ├── naive_bayes.joblib
│   │   ├── random_forest.joblib
│   │   └── xgboost.joblib
│   │
│   ├── results/
│   │   ├── model_comparison.csv
│   │   ├── model_comparison.txt
│   │   ├── model_comparison.html
│   │   └── model_comparison_latex.txt
│   │
│   └── src/
│       ├── model_code/
│       ├── feature_engineering/
│       ├── metrics_generation/
│       ├── visualization.py
│       ├── train.py
│       └── generate_comparison_table.py
```

---

## Deployment

### Deploy on Streamlit Community Cloud

1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit - Adult Census ML Classification"
git branch -M main
git remote add origin https://github.com/your-username/adult-census-ml-classification.git
git push -u origin main
```

2. Deploy App

* Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
* Sign in with GitHub
* Click **New App**
* Select repository
* Branch: `main`
* Main file: `project-folder/app.py`
* Click **Deploy**

3. App URL

```
https://your-username-adult-census-ml.streamlit.app
```

---

## Requirements

### Python Packages

```
streamlit==1.41.1
scikit-learn==1.5.2
xgboost==2.1.3
pandas==2.2.3
numpy==1.26.4
plotly==5.24.1
matplotlib==3.9.3
seaborn==0.13.2
joblib==1.4.2
tabulate==0.9.0
scipy==1.11.4
```

### System Requirements

* RAM: 4GB minimum (8GB recommended)
* Storage: ~500MB
* Python: 3.8+ (tested on 3.11)

---

## Features

### Advanced Feature Engineering

* 51 engineered features from 14 original features
* Binary flags for capital gain/loss
* Log transformations for skewed distributions
* Age and hours-per-week binning
* Interaction and ratio features

### Model Training Features

* Class imbalance handling
* Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
* Stratified train-test split
* Cross-validation (3–5 folds)

### Streamlit App Features

* Interactive UI
* Real-time predictions
* Interactive Plotly visualizations
* Downloadable prediction results

---

## License

This project is for educational purposes as part of **BITS Pilani – ML Assignment 2**.

---

## Author

**ROHITH KRISHNAMURTHY**
BITS Pilani – M.Tech (AIML)
Assignment 2: Machine Learning (AIMLCZG565)

---

## Acknowledgments

* Dataset: UCI Machine Learning Repository
* Extracted by Barry Becker from the 1994 Census database
* Donors: Ronny Kohavi and Barry Becker

---

## Assignment Submission Checklist

* [x] GitHub repository with complete source code
* [x] `requirements.txt` with dependencies
* [x] README with all required sections
* [x] Dataset description (1 mark)
* [x] Comparison table with 6 models (6 marks)
* [x] Model performance observations (3 marks)
* [x] Streamlit app with:

  * Dataset upload
  * Model selection
  * Evaluation metrics
  * Confusion matrix / classification report
* [ ] Screenshot from BITS Virtual Lab (**PENDING**)
* [ ] Deployed Streamlit app link
* [ ] PDF submission with README + screenshot

**Total**: 15 marks (10 + 4 + 1)
**Last Updated**: January 22, 2026
