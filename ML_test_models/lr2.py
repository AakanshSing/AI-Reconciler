import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

# Paths
TRAIN = r"ML_test_models\input\train_v2.csv"
TEST  = r"ML_test_models\input\test__2.csv"
MODEL = r"ML_test_models\lr_match_model2.pkl"
OUT   = r"ML_test_models\output\test__2_output.csv"

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

# 1. Train model if not exists
if not os.path.exists(MODEL):
    train = pd.read_csv(TRAIN)
    train["label"] = train["percentage"].apply(percentage_to_label)
    X = train[FEATURE_COLS]
    y = train["label"]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    clf = LogisticRegression(
        class_weight="balanced", max_iter=1000,
        multi_class="multinomial", solver="lbfgs"
    )
    clf.fit(X_tr, y_tr)

    # Print performance metrics
    prob_val = clf.predict_proba(X_val)
    roc_auc = roc_auc_score(y_val, prob_val, multi_class='ovr')
    print(f"ROC-AUC (OvR): {roc_auc:.4f}")
    print(classification_report(y_val, clf.predict(X_val), digits=3))

    joblib.dump(clf, MODEL)
    print("Trained and saved logistic regression model to", MODEL)
else:
    clf = joblib.load(MODEL)
    print("Loaded existing logistic regression model.")

# 2. Score test set and output match_pct
test = pd.read_csv(TEST)
X_test = test[FEATURE_COLS]
proba = clf.predict_proba(X_test)
test["match_pct"] = (proba[:, 2] * 100).round(2)  # class 2 is 'matcher'

test.to_csv(OUT, index=False)
print("Predictions written to", OUT)
