import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, 
                            precision_score, recall_score, f1_score, 
                            roc_auc_score, precision_recall_curve, roc_curve)
from shap import Explanation

# Load pre-saved model and dataset
@st.cache_data
def load_data():
    # Assuming files are in the same directory as the script
    data_path = Path(__file__).parent / "breast_cancer.csv"
    model_path = Path(__file__).parent / "cancer_model.pkl"
    
    data = pd.read_csv(data_path)
    model = joblib.load(model_path)
    
    # Check for common target column names
    target_col = None
    for col in ['diagnosis', 'target', 'class', 'outcome']:
        if col in data.columns:
            target_col = col
            break
    
    if target_col is None:
        st.error("Could not find target column in dataset. Expected one of: 'diagnosis', 'target', 'class', 'outcome'")
        st.stop()
    
    return data, model, target_col

data, model, target_col = load_data()

# Get feature names from the model
feature_names = model.feature_names_in_
feature_order = list(feature_names)

# Split data into train and test sets (80/20 split)
X = data[feature_names]
y = data[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cache SHAP explainer for faster predictions
@st.cache_resource
def get_shap_explainer(_model):  # Note the underscore prefix
    return shap.TreeExplainer(_model)

explainer = get_shap_explainer(model)  # This will work now

# Predictions on TEST set only (for realistic metrics)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

icon_path = Path(__file__).parent / "icon.png"

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon=icon_path,
    layout="centered",
    initial_sidebar_state="auto",
)

# Streamlit UI
st.title("Breast Cancer Prediction System")

# Sidebar with all features
st.sidebar.header("Patient Features")
inputs = {}
for feature in feature_order:
    readable_name = feature.replace('_', ' ').title()
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        inputs[feature] = st.slider(
            readable_name,
            float(data[feature].min()),
            float(data[feature].max()),
            float(data[feature].mean()),
            step=0.01,
            key=feature
        )
    with col2:
        st.write("")
        st.write(f"Range: {data[feature].min():.1f}-{data[feature].max():.1f}")

# Main content tabs
tab1, tab2, tab3 = st.tabs(["Prediction", "Data Analysis", "Model Info"])

with tab1:
    if st.button("Predict", type="primary"):
        input_df = pd.DataFrame([inputs], columns=feature_order)
        
        # Prediction
        proba = model.predict_proba(input_df)[0][1]
        prediction_class = "Malignant" if proba >= 0.5 else "Benign"
        confidence = proba*100 if prediction_class == "Malignant" else (1-proba)*100

        # Color-coded prediction display
        if prediction_class == "Malignant":
            st.error(f"Prediction: {prediction_class} (Confidence: {confidence:.2f}%)")
        else:
            st.success(f"Prediction: {prediction_class} (Confidence: {confidence:.2f}%)")
        
        # Pie Chart
        fig1, ax1 = plt.subplots()
        ax1.pie(
            [proba, 1-proba],
            labels=['Malignant', 'Benign'],
            colors=['#ff9999','#66b3ff'],
            autopct='%1.1f%%',
            startangle=90
        )
        ax1.axis('equal')
        st.subheader("Prediction Probability")
        st.pyplot(fig1)

        # SHAP Waterfall Plot (using cached explainer)
        st.subheader("Feature Impact Analysis")
        shap_values = explainer(input_df)
        
        exp = Explanation(
            values=shap_values.values[:,:,1],
            base_values=shap_values.base_values[:,1],
            data=input_df.values,
            feature_names=[name.replace('_', ' ').title() for name in feature_order]
        )
        
        plt.figure()
        shap.plots.waterfall(exp[0], max_display=10, show=False)
        st.pyplot(plt.gcf())
        plt.clf()

with tab2:
    # Data Distribution Plots
    st.subheader("Data Distribution")
    
    # Create diagnosis label based on target column
    data['diagnosis_label'] = data[target_col].map({1: 'Malignant', 0: 'Benign'})
    
    # Diagnosis Distribution Pie Chart
    fig2, ax2 = plt.subplots()
    diagnosis_counts = data['diagnosis_label'].value_counts()
    ax2.pie(
        diagnosis_counts,
        labels=diagnosis_counts.index,
        colors=['#66b3ff','#ff9999'],
        autopct='%1.1f%%',
        startangle=90
    )
    ax2.axis('equal')
    st.pyplot(fig2)
    
    # Feature Distribution Boxplot
    st.subheader("Feature Distribution by Diagnosis")
    feature_options = [(feature, feature.replace('_', ' ').title()) for feature in feature_order]
    selected_feature = st.selectbox(
        "Select feature to visualize",
        options=feature_options,
        format_func=lambda x: x[1]
    )
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        x='diagnosis_label',
        y=selected_feature[0],
        data=data,
        palette={'Benign': '#66b3ff', 'Malignant': '#ff9999'},
        width=0.5,
        ax=ax3
    )
    ax3.set_title(f'Distribution of {selected_feature[1]} by Diagnosis')
    ax3.set_xlabel('Diagnosis')
    ax3.set_ylabel(selected_feature[1])
    st.pyplot(fig3)

with tab3:
    # Add selectbox for choosing train or test set
    dataset_choice = st.selectbox(
        "Select dataset to view metrics:",
        ("Train Set", "Test Set"),
        index=0  # Default to Train Set
    )
    
    if dataset_choice == "Train Set":
        # Use training set
        y_set = y_train
        y_set_pred = model.predict(X_train)
        y_set_pred_proba = model.predict_proba(X_train)[:, 1]
        set_name = "Train Set"
    else:
        # Use test set
        y_set = y_test
        y_set_pred = y_pred
        y_set_pred_proba = y_pred_proba
        set_name = "Test Set"
    
    st.subheader(f"Model Performance Metrics ({set_name})")
    
    # Calculate metrics
    precision = precision_score(y_set, y_set_pred)
    recall = recall_score(y_set, y_set_pred)
    f1 = f1_score(y_set, y_set_pred)
    roc_auc = roc_auc_score(y_set, y_set_pred_proba)
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precision", f"{precision:.3f}")
    col2.metric("Recall", f"{recall:.3f}")
    col3.metric("F1 Score", f"{f1:.3f}")
    col4.metric("ROC AUC", f"{roc_auc:.3f}")
    
    # ROC Curve
    st.subheader(f"ROC Curve ({set_name})")
    fpr, tpr, _ = roc_curve(y_set, y_set_pred_proba)
    fig4, ax4 = plt.subplots()
    ax4.plot(fpr, tpr, color='#4e73df', label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax4.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')
    ax4.set_title(f'Receiver Operating Characteristic ({set_name})')
    ax4.legend()
    st.pyplot(fig4)
    
    # Precision-Recall Curve
    st.subheader(f"Precision-Recall Curve ({set_name})")
    precision_curve, recall_curve, _ = precision_recall_curve(y_set, y_set_pred_proba)
    fig5, ax5 = plt.subplots()
    ax5.plot(recall_curve, precision_curve, color='#4e73df', label='Precision-Recall curve')
    ax5.set_xlabel('Recall')
    ax5.set_ylabel('Precision')
    ax5.set_title(f'Precision-Recall Curve ({set_name})')
    ax5.legend()
    st.pyplot(fig5)
    
    # Feature Importance Plot (unchanged - not dataset specific)
    st.subheader("Top 10 Important Features")
    importances = model.feature_importances_
    top_10_indices = np.argsort(importances)[-10:]
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    readable_names = [name.replace('_', ' ').title() for name in feature_order]
    ax6.barh(range(len(top_10_indices)), importances[top_10_indices], color='#4e73df', align='center')
    ax6.set_yticks(range(len(top_10_indices)))
    ax6.set_yticklabels([readable_names[i] for i in top_10_indices])
    ax6.set_xlabel('Relative Importance')
    st.pyplot(fig6)
    
    # Confusion Matrix
    st.subheader(f"Confusion Matrix ({set_name})")
    cm = confusion_matrix(y_set, y_set_pred)
    fig7, ax7 = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=['Benign', 'Malignant']).plot(ax=ax7)
    st.pyplot(fig7)

# Show raw data option
if st.checkbox("Show raw data"):
    st.write(data)