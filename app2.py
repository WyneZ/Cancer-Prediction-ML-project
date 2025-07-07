import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import shap
from shap import Explanation
from pathlib import Path

# Load the model and dataset
model = joblib.load("cancer_model.pkl")
data = pd.read_csv("breast_cancer.csv")

# Feature names
feature_names = model.feature_names_in_

icon_path = Path(__file__).parent / "icon.png"

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon=icon_path,
    layout="centered",
    initial_sidebar_state="auto",
)

st.title(" Breast Cancer Prediction System")

st.markdown("##  Enter Tumor Feature Values Below")

user_input = []
for feature in feature_names:
    value = st.number_input(f"{feature}", min_value=0.0, step=0.1)
    user_input.append(value)

user_input_np = np.array(user_input).reshape(1, -1)

# Threshold slider
threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.5)

if st.button("Predict"):
    probability = model.predict_proba(user_input_np)[0][1]
    prediction = 1 if probability >= threshold else 0

    st.markdown(f"###  Prediction: {'Malignant' if prediction == 1 else 'Benign'}")
    st.markdown(f"###  Confidence Score: `{probability:.2f}`")

    # SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(user_input_np)
    
    st.subheader("🔬 SHAP Feature Impact")
    plt.figure()
    
    # Create proper Explanation object for the positive class
    exp = Explanation(
        values=shap_values.values[:,:,1],
        base_values=shap_values.base_values[:,1],
        data=user_input_np,
        feature_names=feature_names
    )
    
    # Plot waterfall for the first sample
    shap.plots.waterfall(exp[0], max_display=10, show=False)
    st.pyplot(plt.gcf())
    plt.clf()

# ------------------ Model Performance Section ------------------
st.header("Model Performance Metrics")

# Test set (for demo, using part of the dataset as test)
X = data[feature_names]
y = data['target']
y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1]

# Confusion matrix
cm = confusion_matrix(y, y_pred)
st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.xlabel("Predicted label")
plt.ylabel("True label")
st.pyplot(fig)

# Metrics
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
roc = roc_auc_score(y, y_proba)

st.write("**Precision:**", round(precision, 2))
st.write("**Recall:**", round(recall, 2))
st.write("**F1 Score:**", round(f1, 2))
st.write("**ROC AUC Score:**", round(roc, 2))

# Feature Importance
st.subheader("Top 10 Important Features")
importances = model.feature_importances_
sorted_idx = np.argsort(importances)[::-1][:10]
plt.figure(figsize=(8, 5))
plt.barh(np.array(feature_names)[sorted_idx][::-1], importances[sorted_idx][::-1], color='skyblue')
plt.xlabel("Relative Importance")
st.pyplot(plt.gcf())