"""Train 5 classifiers on one dataset, save pipelines + test data + metrics.

Edit the CONFIG block, then run:  python train_models.py
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ----------------------------- CONFIG ---------------------------------
DATA_PATH = "dry_bean.csv"        # <-- your downloaded dataset file
TARGET = "Class"             # <-- name of the label column
DROP_COLS = []                # <-- id / useless columns, e.g. ["id", "Unnamed: 32"]
TEST_SIZE = 0.2
SEED = 42
# ----------------------------------------------------------------------

os.makedirs("model", exist_ok=True)

df = pd.read_csv(DATA_PATH)
df = df.drop(columns=DROP_COLS, errors="ignore").dropna(subset=[TARGET])
print("Shape:", df.shape, "| classes:", df[TARGET].nunique())

X = df.drop(columns=[TARGET])
y_raw = df[TARGET]
le = LabelEncoder()
y = le.fit_transform(y_raw)
joblib.dump(le, "model/label_encoder.joblib")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)

# save raw (un-encoded) test data incl. target for the Streamlit app
test_df = X_test.copy()
test_df[TARGET] = le.inverse_transform(y_test)
test_df.to_csv("test_data.csv", index=False)

num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

preprocess = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore",
                                           sparse_output=False))]), cat_cols),
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=SEED),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(kernel="rbf", probability=True,
                                  random_state=SEED),
}

n_classes = len(le.classes_)
avg = "binary" if n_classes == 2 else "weighted"


def evaluate(model, X_te, y_te):
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)
    if n_classes == 2:
        auc = roc_auc_score(y_te, proba[:, 1])
    else:
        auc = roc_auc_score(y_te, proba, multi_class="ovr", average="weighted")
    return {
        "Accuracy": accuracy_score(y_te, pred),
        "AUC": auc,
        "Precision": precision_score(y_te, pred, average=avg, zero_division=0),
        "Recall": recall_score(y_te, pred, average=avg, zero_division=0),
        "F1": f1_score(y_te, pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_te, pred),
    }


rows = {}
for name, clf in models.items():
    pipe = Pipeline([("prep", preprocess), ("clf", clf)])
    pipe.fit(X_train, y_train)
    rows[name] = evaluate(pipe, X_test, y_test)
    fname = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(pipe, os.path.join("model", fname))
    print(f"done: {name}")

res = pd.DataFrame(rows).T.round(4)
res.index.name = "ML Model Name"
res.to_csv("model/results.csv")
json.dump({"target": TARGET, "classes": [str(c) for c in le.classes_]},
          open("model/meta.json", "w"))
print("\n", res.to_string())
