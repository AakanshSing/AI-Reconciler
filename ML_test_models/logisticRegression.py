import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

# Paths
TRAIN = r"ML_test_models\input\train_v2_2.csv"
TEST  = r"ML_test_models\input\test__2.csv"
MODEL = r"ML_test_models\lr_match_model4.pkl"
OUT   = r"ML_test_models\output\test__4_output.csv"

FEATURE_COLS = [
    "customer_id_binary", "policy_no_binary", "name_binary",
    "location_binary", "product_binary", "channel_binary", "amount_pct"
]

def percentage_to_label(perc):
    if perc > 95:
        return 2    # matcher
    elif perc >= 80:
        return 1    # checker
    else:
        return 0    # unmatched

# bump params
A = 0.02
t = 0.95
k = 50

# 1. train or load base LR
if not os.path.exists(MODEL):
    df = pd.read_csv(TRAIN)
    df["label"] = df["percentage"].apply(percentage_to_label)
    X, y = df[FEATURE_COLS], df["label"]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    clf = LogisticRegression(
        class_weight="balanced", max_iter=1000,
        multi_class="multinomial", solver="lbfgs"
    )
    clf.fit(X_tr, y_tr)
    joblib.dump(clf, MODEL)
else:
    clf = joblib.load(MODEL)

# 2. score & collapse
test = pd.read_csv(TEST)
proba = clf.predict_proba(test[FEATURE_COLS])
m_raw = proba[:,1] + proba[:,2]      # collapsed prob in [0,1]

# 3. smooth bump
m_bumped = m_raw - A / (1 + np.exp(-k*(m_raw - t)))

# 4. attach results
test["match_pct"] = (m_bumped * 100).round(2)
test["match"]     = (m_bumped >= 0.75).astype(int)

# 5. evaluation (requires TEST to still have `percentage`)
if "percentage" in test.columns:
    # -- multiclass metrics --
    y_true_mc = test["percentage"].apply(percentage_to_label)
    roc_mc = roc_auc_score(y_true_mc, proba, multi_class="ovr")
    print(f"Multiclass ROC-AUC (OvR): {roc_mc:.4f}")
    y_pred_mc = clf.predict(test[FEATURE_COLS])
    print("Multiclass Classification Report:")
    print(classification_report(y_true_mc, y_pred_mc, digits=3))

    # -- binary metrics (classes 1+2 vs 0) --
    y_true_bin = (y_true_mc >= 1).astype(int)
    roc_bin = roc_auc_score(y_true_bin, m_bumped)
    print(f"Binary ROC-AUC after bump: {roc_bin:.4f}")
    print("Binary Classification Report (match vs no-match):")
    print(classification_report(y_true_bin, test["match"], digits=3))

# 6. save
test.to_csv(OUT, index=False)
print("Predictions written to", OUT)
