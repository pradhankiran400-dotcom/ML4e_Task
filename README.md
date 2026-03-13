<p>
  <h1>Kiran Kumar Pradhan</h1>
  <h2>125ID0012</h2>
</p>
# Machine Learning Projects

Two machine learning projects covering classification and regression on real-world datasets.

---

## Repository Structure

```
├── Task1/
│   ├── nyc_housing_base.csv
│   └── Regression_on_Housing_dataset.ipynb
|   └── LinearRegression_From_Scratch.docx
│
├── Task-2/
│   ├── diabetes.csv
│   └── Classification_on_Diabetes_dataset (1).ipynb
|   └── ML4e_Task2_document.docx
│
└── README.md
```

---

## Project 1 — NYC Housing Price Prediction and  and Logistic Regression from Scratch

Predicting property sale prices using Linear Regression implemented from scratch in pure NumPy — no scikit-learn model used.

### Dataset
NYC housing records with features: building_age, landuse, borough_x, block. Target: sale_price. Preprocessing includes duplicate removal, missing value imputation, IQR outlier removal, and StandardScaler normalisation.

### Models Built from Scratch

**MYLR (Simple Linear Regression)** — works with a single feature. Uses the Ordinary Least Squares closed-form formula to find the exact slope and intercept in one pass with no iteration.

**MYLR_Multiple (Multiple Linear Regression)** — works with all four features. Uses Gradient Descent to iteratively update one weight per feature and a bias term over 1000 epochs, minimising Mean Squared Error.

| | MYLR | MYLR_Multiple |
|---|---|---|
| Features | Single only | Any number |
| Method | OLS — exact formula | Gradient Descent |
| Iterations | None | 1000 epochs |
| Used for housing data | No | Yes |

---

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---
## Project 2 — Diabetes Prediction and Linear Regression from Scratch

Predicting whether a patient is diabetic or non-diabetic using three classification algorithms.

### Dataset
764 patient records with features: Glucose, BMI, Age. Target: Outcome (1 = Diabetic, 0 = Healthy).

### Models

**Logistic Regression** — computes a probability using a sigmoid function. If probability ≥ 0.5, the patient is classified as diabetic. Accuracy ~77%.

**Decision Tree** — splits data using Gini Impurity. The first split is always on Glucose ≤ 127.5 because glucose has the strongest relationship with diabetes. Accuracy ~72%.

**Random Forest** — builds 100 decision trees and takes majority vote, reducing overfitting compared to a single tree. Accuracy ~80%.

### Results

| Model | Accuracy | Recall |
|---|---|---|
| Logistic Regression | ~77% | ~63% |
| Decision Tree | ~72% | ~61% |
| Random Forest | ~80% | ~67% |

### Confusion Matrix

| | Predicted Healthy | Predicted Diabetic |
|---|---|---|
| Actual Healthy | 78 ✓ | 21 ✗ |
| Actual Diabetic | 18 ✗ ⚠️ | 37 ✓ |

The 18 false negatives (diabetic patients predicted as healthy) are the most critical error — in medical prediction, Recall matters more than Accuracy.

---


