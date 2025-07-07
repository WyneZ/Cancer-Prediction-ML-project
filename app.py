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

# Standardize target encoding (convert strings to numbers if needed)
if data[target_col].dtype == 'object':
    # Convert common string encodings to numerical
    data[target_col] = data[target_col].map({'M': 1, 'B': 0, 'malignant': 1, 'benign': 0}).astype(int)
else:
    # Ensure numerical encoding is correct
    unique_values = sorted(data[target_col].unique())
    if len(unique_values) != 2:
        st.error(f"Target column must have exactly 2 classes, found {len(unique_values)}")
        st.stop()
    
    # Assume the higher value is malignant (standard convention)
    if unique_values != [0, 1]:
        st.warning(f"Target column has unexpected values {unique_values}. Mapping {max(unique_values)} to Malignant.")
        data[target_col] = data[target_col].map({max(unique_values): 1, min(unique_values): 0})

# Create diagnosis labels (now guaranteed correct)
data['diagnosis_label'] = data[target_col].map({1: 'Benign', 0: 'Malignant'})


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
        prediction_class = "Benign" if proba >= 0.5 else "Malignant"
        confidence = proba*100 if prediction_class == "Benign" else (1-proba)*100

        # Color-coded prediction display
        if prediction_class == "Malignant":
            st.error(f"Prediction: {prediction_class} (Confidence: {confidence:.2f}%)")
        else:
            st.success(f"Prediction: {prediction_class} (Confidence: {confidence:.2f}%)")
        
        # Pie Chart
        fig1, ax1 = plt.subplots()
        ax1.pie(
            [proba, 1-proba],
            labels=['Benign', 'Malignant'],
            colors=['#66b3ff','#ff9999'],
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

        # Explanation for SHAP plot
        with st.expander("ℹ️ Understanding Feature Impact"):
            st.markdown("""
            **How to read this chart**:
            - Features pushing the prediction **right** (red) increase malignancy likelihood
            - Features pushing **left** (blue) decrease malignancy likelihood
            - Larger bars = stronger impact on the prediction
            - Base value: Average prediction before considering these features
            """)

with tab2:
    # Data Distribution Plots
    st.subheader("Data Distribution")
    
    # Diagnosis Distribution Pie Chart
    fig2, ax2 = plt.subplots()
    diagnosis_counts = data['diagnosis_label'].value_counts()
    
    # Ensure the order is Benign first, then Malignant
    diagnosis_counts = diagnosis_counts.reindex(['Benign', 'Malignant'])
    
    ax2.pie(
        diagnosis_counts,
        labels=diagnosis_counts.index,
        colors=['#66b3ff','#ff9999'],
        autopct='%1.1f%%',
        startangle=90
    )
    ax2.axis('equal')
    st.pyplot(fig2)
    
    # Explanation for Data Distribution
    st.caption(f"""
    **Clinical Context**:  
    - This shows the proportion of benign vs malignant cases in our dataset.  
    - A balanced dataset (close to 50-50) helps the model learn both classes equally.  
    - Current distribution: **{diagnosis_counts['Benign']} benign** ({diagnosis_counts['Benign']/sum(diagnosis_counts)*100:.1f}%) vs **{diagnosis_counts['Malignant']} malignant** ({diagnosis_counts['Malignant']/sum(diagnosis_counts)*100:.1f}%).
    """)
    
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
    accuracy = (y_set == y_set_pred).mean()
    
    # Display metrics in columns with explanations
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precision", f"{precision:.3f}", 
               help="% of malignant predictions that were correct")
    col2.metric("Recall", f"{recall:.3f}", 
               help="% of actual malignancies detected")
    col3.metric("F1 Score", f"{f1:.3f}", 
               help="Balance between precision and recall")
    col4.metric("ROC AUC", f"{roc_auc:.3f}", 
               help="Overall classification ability (1.0 = perfect)")
    
    # Detailed metric explanations
    with st.expander("ℹ️ Understanding These Metrics"):
        st.markdown(f"""
        ### Clinical Performance Interpretation:
        - **Precision ({precision*100:.1f}%)**:  
          When the model predicts malignant, it's correct **{precision*100:.1f}%** of the time.  
          *Higher values mean fewer false alarms (unnecessary biopsies).*
          
        - **Recall/Sensitivity ({recall*100:.1f}%)**:  
          The model detects **{recall*100:.1f}%** of actual malignant cases.  
          *Higher values mean fewer missed cancers (critical for early detection).*
          
        - **F1 Score ({f1:.3f})**:  
          Balanced measure considering both false positives and negatives.  
          *Values closer to 1.0 indicate better overall performance.*
          
        - **ROC AUC ({roc_auc:.3f})**:  
          - 0.9-1.0 = Excellent discrimination  
          - 0.8-0.9 = Good  
          - 0.7-0.8 = Fair  
          - <0.7 = Poor  
        """)
    
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
    
    # ROC Curve explanation
    st.caption("""
    **Clinical Guidance**:  
    - The curve shows the trade-off between sensitivity (true positive rate) and specificity (1 - false positive rate)  
    - Ideal curve hugs the top-left corner (high sensitivity with few false positives)  
    - The diagonal line represents random guessing  
    """)
    
    # Precision-Recall Curve
    st.subheader(f"Precision-Recall Curve ({set_name})")
    precision_curve, recall_curve, _ = precision_recall_curve(y_set, y_set_pred_proba)
    fig5, ax5 = plt.subplots()
    ax5.plot(recall_curve, precision_curve, color='#4e73df', label='Precision-Recall curve')
    ax5.set_xlabel('Recall (Sensitivity)')
    ax5.set_ylabel('Precision')
    ax5.set_title(f'Precision-Recall Curve ({set_name})')
    ax5.legend()
    st.pyplot(fig5)
    
    # Precision-Recall explanation
    st.caption("""
    **Clinical Decision Making**:  
    - Shows how precision changes as we adjust sensitivity  
    - Steep drops indicate thresholds where false positives increase rapidly  
    - Helps select operating points based on clinical priorities  
    """)
    
    # Feature Importance Plot
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
    
    # Feature importance explanation
    st.caption("""
    **Clinical Relevance**:  
    - Shows which features most influence predictions  
    - Helps validate if medically important factors are being weighted appropriately  
    - Does NOT imply causation - always consider clinical context  
    """)
    
    # Confusion Matrix
    st.subheader(f"Confusion Matrix ({set_name})")
    cm = confusion_matrix(y_set, y_set_pred)
    fig7, ax7 = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=['Benign', 'Malignant']).plot(ax=ax7)
    st.pyplot(fig7)
    
    # Confusion matrix explanation
    with st.expander("ℹ️ How to Interpret This Matrix"):
        st.markdown(f"""
        **Clinical Impact Analysis**:  
        - **True Negatives ({cm[0,0]})**: Correct benign diagnoses  
        - **False Positives ({cm[0,1]})**: Healthy cases flagged as malignant  
          *May cause patient anxiety/unnecessary procedures*  
        - **False Negatives ({cm[1,0]})**: Missed malignancies  
          *Clinically dangerous - prioritize minimizing these*  
        - **True Positives ({cm[1,1]})**: Correct cancer detections  
        
        **Current Error Rates**:  
        - False alarm rate: {cm[0,1]/(cm[0,0]+cm[0,1])*100:.1f}% of benign cases  
        - Missed cancer rate: {cm[1,0]/(cm[1,0]+cm[1,1])*100:.1f}% of malignant cases  
        """)

# Show raw data option
if st.checkbox("Show raw data"):
    st.write(data)
    st.caption("""
    **Note to Clinicians**:  
    - Review feature distributions for clinical plausibility  
    - Check for unexpected values that may affect model performance  
    - All values should be within medically reasonable ranges  
    """)