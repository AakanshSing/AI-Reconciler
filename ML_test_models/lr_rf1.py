# lr.py
"""
Reconciliation matcher pipeline with Logistic Regression & Random Forest baselines.

Schema:
  ML_test_models/
    ├ input/
    │   ├ train_v2.csv       # feature-only dataset with 'percentage'
    │   └ feature_test.csv   # feature-only test dataset (no label)
    ├ output/
    └ lr.py  # this script

Usage:
    python lr.py

Outputs:
    ML_test_models/lr_match_model2.pkl      # trained LogisticRegression model
    ML_test_models/rf_match_model.pkl       # trained RandomForest model
    ML_test_models/output/test_feature_predictions.csv  # test + match_pct_lr + match_pct_rf

Dependencies:
    pandas, numpy, scikit-learn, joblib, os
"""
from __future__ import annotations
import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths (hard-coded)
# ---------------------------------------------------------------------------
TRAIN_CSV = r"ML_test_models\input\train_v2.csv"
TEST_CSV  = r"ML_test_models\input\feature_test.csv"
LR_MODEL  = r"ML_test_models\lr_match_model2.pkl"
RF_MODEL  = r"ML_test_models\rf_match_model.pkl"
PRED_OUT  = r"ML_test_models\output\test_feature_predictions_lr_rf.csv"

# ---------------------------------------------------------------------------
# Feature & label configuration
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "customer_binary", "Policy_binary", "name_binary",
    "location_binary", "product_binary", "channel_binary", "amount_diff"
]

# Discretize continuous percentage into three classes
def percentage_to_label(perc: float) -> int:
    if perc > 95:
        return 2    # matcher
    elif perc >= 80:
        return 1    # checker
    else:
        return 0    # unmatched

# ---------------------------------------------------------------------------
# Load train data, create labels, split
# ---------------------------------------------------------------------------
df_train = pd.read_csv(TRAIN_CSV)
if "label" not in df_train.columns:
    df_train["label"] = df_train["percentage"].apply(percentage_to_label)
X = df_train[FEATURE_COLS]
y = df_train["label"]
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------------------------------------------------------------------
# 1) Logistic Regression baseline
# ---------------------------------------------------------------------------
clf_lr = None
if os.path.exists(LR_MODEL):
    clf_lr = joblib.load(LR_MODEL)
    print(f"Loaded LR model from {LR_MODEL}")
else:
    clf_lr = LogisticRegression(
        class_weight="balanced", max_iter=1000,
        multi_class="multinomial", solver="lbfgs"
    )
    clf_lr.fit(X_tr, y_tr)
    joblib.dump(clf_lr, LR_MODEL)
    print(f"Trained LR model saved to {LR_MODEL}")

prob_val_lr = clf_lr.predict_proba(X_val)
print("\n=== Logistic Regression Metrics ===")
print("ROC-AUC (OvR):", round(roc_auc_score(y_val, prob_val_lr, multi_class="ovr"),4))
print(classification_report(y_val, clf_lr.predict(X_val), target_names=["unmatched","checker","matcher"]))

# ---------------------------------------------------------------------------
# 2) Random Forest baseline
# ---------------------------------------------------------------------------
clf_rf = None
if os.path.exists(RF_MODEL):
    clf_rf = joblib.load(RF_MODEL)
    print(f"Loaded RF model from {RF_MODEL}")
else:
    clf_rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced_subsample",
        random_state=42
    )
    clf_rf.fit(X_tr, y_tr)
    joblib.dump(clf_rf, RF_MODEL)
    print(f"Trained RF model saved to {RF_MODEL}")

prob_val_rf = clf_rf.predict_proba(X_val)
print("\n=== Random Forest Metrics ===")
print("ROC-AUC (OvR):", round(roc_auc_score(y_val, prob_val_rf, multi_class="ovr"),4))
print(classification_report(y_val, clf_rf.predict(X_val), target_names=["unmatched","checker","matcher"]))

# ---------------------------------------------------------------------------
# 3) Score test set with both models
# ---------------------------------------------------------------------------
df_test = pd.read_csv(TEST_CSV)
X_test = df_test[FEATURE_COLS]

# LR predictions
match_pct_lr = clf_lr.predict_proba(X_test)[:, 2] * 100  # class 2 = matcher
# RF predictions
match_pct_rf = clf_rf.predict_proba(X_test)[:, 2] * 100

df_test["match_pct_lr"] = match_pct_lr.round(2)
df_test["match_pct_rf"] = match_pct_rf.round(2)

df_test.to_csv(PRED_OUT, index=False)
print(f"Test predictions written to {PRED_OUT}")
