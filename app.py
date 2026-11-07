import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

st.set_page_config(page_title="Classifier Lab", page_icon="🧪", layout="wide")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}


@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_resource
def load_meta():
    le = joblib.load("model/label_encoder.joblib")
    meta = json.load(open("model/meta.json"))
    return le, meta


def compute_metrics(model, X, y, n_classes):
    pred = model.predict(X)
    proba = model.predict_proba(X)
    avg = "binary" if n_classes == 2 else "weighted"
    if n_classes == 2:
        auc = roc_auc_score(y, proba[:, 1])
    else:
        auc = roc_auc_score(y, proba, multi_class="ovr", average="weighted",
                            labels=np.arange(n_classes))
    return pred, {
        "Accuracy": accuracy_score(y, pred),
        "AUC": auc,
        "Precision": precision_score(y, pred, average=avg, zero_division=0),
        "Recall": recall_score(y, pred, average=avg, zero_division=0),
        "F1": f1_score(y, pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }


le, meta = load_meta()
TARGET = meta["target"]
n_classes = len(le.classes_)

st.title("🧪 Classifier Lab")
st.caption(f"Pre-trained models predicting **{TARGET}** "
           f"({n_classes} classes). Upload test data to evaluate - "
           "nothing is retrained here.")

with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload test CSV", type="csv")
    choice = st.selectbox("Choose a model", list(MODEL_FILES))
    show_all = st.checkbox("Compare all 5 models", value=True)

if uploaded is None:
    st.info("Upload a CSV (all feature columns + the target column "
            f"`{TARGET}`) from the sidebar to begin.")
    st.stop()

data = pd.read_csv(uploaded)
if TARGET not in data.columns:
    st.error(f"Target column `{TARGET}` not found in the uploaded file.")
    st.stop()

st.subheader("Data preview")
st.write(f"{data.shape[0]} rows x {data.shape[1]} columns")
st.dataframe(data.head(10), use_container_width=True)

X = data.drop(columns=[TARGET])
try:
    y = le.transform(data[TARGET])
except ValueError:
    st.error("Target column has labels the models were not trained on.")
    st.stop()

model = load_model(MODEL_FILES[choice])
pred, metrics = compute_metrics(model, X, y, n_classes)

st.subheader(f"Metrics - {choice}")
cols = st.columns(6)
for c, (k, v) in zip(cols, metrics.items()):
    c.metric(k, f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    cm = confusion_matrix(y, pred, labels=np.arange(n_classes))
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.markdown("**Classification report**")
    rep = classification_report(y, pred, target_names=[str(c) for c in le.classes_],
                                output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.round(3), use_container_width=True)

if show_all:
    st.subheader("All models on this data")
    rows = {}
    for name, path in MODEL_FILES.items():
        _, m = compute_metrics(load_model(path), X, y, n_classes)
        rows[name] = m
    st.dataframe(pd.DataFrame(rows).T.round(4), use_container_width=True)
