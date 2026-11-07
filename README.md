# Dry Bean Classification - ML Assignment 2

**Live app:** <PASTE STREAMLIT LINK HERE>
**Repository:** https://github.com/pratik-bits-2026aj05/project-assignment-2-ml

## a. Problem statement
Classify dry beans into one of seven varieties (Barbunya, Bombay, Cali, Dermason, Horoz, Seker, Sira) using geometric measurements taken from images of the beans. Five classifiers are trained on the same data, compared on six metrics, and exposed through a Streamlit app that evaluates the saved models on an uploaded test CSV.

## b. Dataset description
- Source: Dry Bean Dataset (UCI Machine Learning Repository / Kaggle)
- Instances: 13,611 (minimum required: 500)
- Features: 16 numeric shape descriptors (minimum required: 12), e.g. Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness, ShapeFactor1-4
- Target: `Class` (7 classes, multi-class problem; imbalanced, Bombay is the smallest with 522 rows, Dermason the largest with 3,546)
- Missing values: none
- Split: 80% train / 20% test (2,723 rows), stratified, random_state=42. The test split is saved as `test_data.csv`.
- Preprocessing: median imputation and standard scaling for numeric features, wrapped in a scikit-learn Pipeline together with each model.

## c. GitHub repository
https://github.com/pratik-bits-2026aj05/project-assignment-2-ml

Contents: `app.py`, `train_models.py`, `requirements.txt`, `README.md`, `test_data.csv`, `dry_bean.csv`, `model/` (five `.joblib` pipelines, label encoder, results).

## d. Models used

Metrics are computed on the held-out test set. For this 7-class problem Precision, Recall and F1 are weighted averages, and AUC is one-vs-rest (weighted).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9214 | 0.9934 | 0.9222 | 0.9214 | 0.9216 | 0.9050 |
| Decision Tree | 0.9041 | 0.9789 | 0.9038 | 0.9041 | 0.9038 | 0.8841 |
| KNN | 0.9137 | 0.9837 | 0.9144 | 0.9137 | 0.9139 | 0.8956 |
| Naive Bayes | 0.8979 | 0.9902 | 0.9007 | 0.8979 | 0.8981 | 0.8773 |
| Support Vector Machine | 0.9221 | 0.9937 | 0.9224 | 0.9221 | 0.9222 | 0.9059 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very strong for a linear model (92.1% accuracy, AUC 0.993), which suggests the classes are close to linearly separable once features are scaled. Its main errors are Dermason predicted as Sira. |
| Decision Tree | Lowest AUC (0.979) and weak on Sira (recall 0.825) and Barbunya (0.853). A single depth-limited tree gives coarse probabilities and splits unevenly on overlapping shape features. Easy to interpret, but less accurate. |
| KNN | Good (91.4%) and better than the tree, as scaled distance in shape space groups similar beans. Slightly behind Logistic Regression and SVM, and predictions are slower because the training data is stored. |
| Naive Bayes | Lowest accuracy (89.8%) and MCC. Many features such as Area, Perimeter and ConvexArea are strongly correlated, which breaks the independence assumption. Its AUC is still high (0.990), so the probability ranking is good even when the hard labels are not. Weakest on Barbunya and Dermason. |
| Support Vector Machine | Best overall (92.2% accuracy, F1 0.922, MCC 0.906). The RBF kernel handles the overlap between similar classes slightly better than the others, but it is only marginally ahead of Logistic Regression. |
| **Overall winner** | **Support Vector Machine** by a very small margin over Logistic Regression. All models separate Bombay perfectly (recall 1.0); the largest remaining error in every model is between Dermason and Sira, which have similar shapes. |

## How to run
```
pip install -r requirements.txt
python train_models.py        # trains the 5 models, writes model/ and test_data.csv
streamlit run app.py
```
Upload `test_data.csv` in the app, choose a model, and view the metrics, confusion matrix and classification report. The app loads the saved models and does not retrain.
