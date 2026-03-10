# Chronic Kidney Disease Prediction
**MSc Artificial Intelligence — Portfolio Project**

---

## About the Data

The dataset contains 1,659 patient records with 51 features and a binary target — whether a patient has CKD or not. It covers a broad range of clinical measurements: kidney function biomarkers, metabolic panels, symptom severity scores, lifestyle factors, and medication history. There are no missing values.

The biggest issue with this dataset is class imbalance — 91.9% of patients are diagnosed with CKD, leaving only 135 non-CKD cases. This makes accuracy a useless metric. A model that predicts CKD for every single patient scores 91.9% without learning anything, so I focused on ROC-AUC and Macro F1 instead.

---

## Methods

**Feature Selection**

Starting with 51 features, I used permutation importance on a held-out test set to rank them. The idea is simple — shuffle each feature's values and measure how much AUC drops. If shuffling doesn't hurt performance, the feature carries no real signal. 32 out of 51 features had zero or negative permutation importance, so they were dropped. The remaining 10 features improved ROC-AUC from 0.77 to 0.84.

I chose permutation importance over built-in GBM importance because built-in scores just measure how often a feature appears in splits — not whether it actually helps on unseen data.

**Preprocessing**

The pipeline is straightforward: select the 10 features, apply StandardScaler (fit on training data only), then oversample the minority class to balance the training set. The oversampling happens strictly after the train/test split — doing it before would mean inflated copies of test patients leaking into training.

**Class Imbalance**

Random oversampling of the minority class was used to balance training. I also evaluated SMOTE but it didn't meaningfully improve minority class recall given the small number of non-CKD cases (135 patients).

---

## Model

I compared Logistic Regression, Random Forest, and Gradient Boosting. GBM worked best here — it iteratively focuses on misclassified samples, which naturally helps after oversampling, and shallow trees act as regularisation. Final hyperparameters were chosen by grid search:

```
GradientBoostingClassifier(
    n_estimators=100, max_depth=4,
    learning_rate=0.05, subsample=0.8
)
```

---

## Findings

| Metric | All 51 features | Top 10 features |
|---|---|---|
| ROC-AUC | 0.766 | **0.829** |
| Macro F1 | 0.664 | **0.676** |
| Recall (No CKD) | 0.41 | **0.48** |

Removing 41 noisy features improved every meaningful metric. The two most important features were SerumCreatinine and GFR, which makes clinical sense — they are the standard diagnostic markers for kidney function. Symptom scores (muscle cramps, itching, nausea) also contributed meaningfully.

The main limitation is the minority class. Even with oversampling, recall for non-CKD patients only reached 0.48 — the model still misses roughly half of true negative cases. With only 135 non-CKD patients in the dataset, there simply isn't enough data to learn a reliable boundary for that class.

---

## Running the Project

```bash
pip install -r requirements.txt
python train_model.py
uvicorn main:app --reload
```

---

*Dataset: Chronic Kidney Disease — Kaggle (1,659 patients, 51 features)*
