# lr.py
"""
Simple reconciliation matcher pipeline with hard-coded paths.

Schema:
  ML_test_models/
    ├ input/
    │   ├ train.csv  # either raw BS/PR file or pre-featurized dataset
    │   └ test2.csv  # raw BS/PR file
    ├ output/
    └ lr.py  # this script

Usage:
  Just `python lr.py`. No arguments needed.

Outputs:
  output/lr_match_model.pkl        # trained LogisticRegression model
  output/test_predictions.csv      # test rows + match_pct + feature columns

Logic:
  - If the train.csv contains feature columns (customer_binary, Policy_binary, etc.),
    they are used directly.
  - Otherwise, raw BS/PR columns are auto-converted by the feature-engineering step.

Required for raw mode:
  Customer ID_BS/PR, Policy No_BS/PR,
  Product_BS/PR, channel_BS/PR, Amount_BS/PR,
  name_binary, location_binary

Dependencies:
  pandas, numpy, scikit-learn, joblib
"""
from __future__ import annotations
import re
import joblib
from typing import Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths (hard-coded)
# ---------------------------------------------------------------------------
TRAIN_CSV = r"ML_test_models\input\train.csv"
TEST_CSV  = r"ML_test_models\input\test2.csv"
MODEL_OUT = r"ML_test_models\lr_match_model.pkl"
PRED_OUT  = r"ML_test_models\output\test_predictions.csv"

# ---------------------------------------------------------------------------
# Constants & mapping
# ---------------------------------------------------------------------------
_UNMATCHED  = 0
_MATCHED    = 1
_MISSING    = -1
_AMOUNT_RE  = re.compile(r"[^0-9.\-]")
LABEL_MAP   = {"unmatched": 0, "checker": 1, "matcher": 2}

# ---------------------------------------------------------------------------
# Feature-engineering utils
# ---------------------------------------------------------------------------

def _is_missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and v.strip() == "")


def _norm(v: Any) -> str:
    return "" if _is_missing(v) else str(v).strip().lower()


def compare_ids(a: Any, b: Any) -> int:
    if _is_missing(a) or _is_missing(b):
        return _MISSING
    return _MATCHED if _norm(a) == _norm(b) else _UNMATCHED


def compare_text(a: Any, b: Any) -> int:
    return compare_ids(a, b)


def _to_float(x: Any) -> float | None:
    if _is_missing(x):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(_AMOUNT_RE.sub("", str(x)))
    except ValueError:
        return None


def amount_diff(a: Any, b: Any) -> float:
    fa, fb = _to_float(a), _to_float(b)
    if fa is None or fb is None or max(abs(fa), abs(fb)) == 0:
        return 1.0
    return round(abs(fa - fb) / max(abs(fa), abs(fb)), 3)

# ---------------------------------------------------------------------------
# Build features from raw BS/PR
# ---------------------------------------------------------------------------
_FIELD_MAP = {
    "customer_binary": ("Customer ID_BS", "Customer ID_PR", compare_ids),
    "Policy_binary":   ("Policy No_BS",   "Policy No_PR",   compare_ids),
    "product_binary":  ("Product_BS",     "Product_PR",     compare_text),
    "channel_binary":  ("channel_BS",     "channel_PR",     compare_text),
}
_AMOUNT_COLS = ("Amount_BS", "Amount_PR")


def build_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    feat = {k: [] for k in _FIELD_MAP}
    feat["amount_diff"] = []

    for _, row in df.iterrows():
        for name, (lcol, rcol, fn) in _FIELD_MAP.items():
            feat[name].append(fn(row.get(lcol), row.get(rcol)))
        feat["amount_diff"].append(amount_diff(row.get(_AMOUNT_COLS[0]), row.get(_AMOUNT_COLS[1])))

    for col in ("name_binary", "location_binary"):
        if col not in df.columns:
            raise KeyError(f"Missing column: {col}")
        feat[col] = df[col].tolist()

    return pd.DataFrame(feat)

# ---------------------------------------------------------------------------
# Train & score pipeline
# ---------------------------------------------------------------------------

def train_and_save():
    df = pd.read_csv(TRAIN_CSV)
    # Feature detection
    if "customer_binary" in df.columns:
        feature_cols = [* _FIELD_MAP.keys(), "amount_diff", "name_binary", "location_binary"]
        X = df[feature_cols]
    else:
        X = build_feature_set(df)

    y = df["label"].map(LABEL_MAP)
    if y.isnull().any():
        raise KeyError("Training CSV must contain 'label' column with valid values.")

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(X_tr, y_tr)

    prob = clf.predict_proba(X_val)
    print(f"ROC-AUC (OvR): {roc_auc_score(y_val, prob, multi_class='ovr'):.4f}")

    joblib.dump(clf, MODEL_OUT)
    print(f"Model saved to {MODEL_OUT}")
    return clf


def score_and_save(clf):
    df = pd.read_csv(TEST_CSV)
    if "customer_binary" in df.columns:
        feature_cols = [* _FIELD_MAP.keys(), "amount_diff", "name_binary", "location_binary"]
        X = df[feature_cols]
    else:
        X = build_feature_set(df)

    # Add features to output
    for col in X.columns:
        df[col] = X[col].values

    probs = clf.predict_proba(X)
    df["match_pct"] = (probs[:, LABEL_MAP["matcher"]] * 100).round(2)

    df.to_csv(PRED_OUT, index=False)
    print(f"Predictions written to {PRED_OUT}")

if __name__ == "__main__":
    clf = train_and_save()
    score_and_save(clf)
