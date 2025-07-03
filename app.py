import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Load and prepare data
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    columns = ["id", "diagnosis"] + [f"feature_{i}" for i in range(1, 31)]
    data = pd.read_csv(url, header=None, names=columns)
    data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})
    return data

data = load_data()
X = data.drop(['id', 'diagnosis'], axis=1)
y = data['diagnosis']

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

# 3. Train model
@st.cache_resource
def train_model():
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model

model = train_model()

# 4. Streamlit UI
st.title("Breast Cancer Prediction System")

pred = model.predict(X)[0]

# Sidebar inputs
st.sidebar.header("Patient Features")
inputs = {}
for feature in X.columns:
    inputs[feature] = st.sidebar.slider(
        feature,
        float(X[feature].min()),
        float(X[feature].max()),
        float(X[feature].mean())
    )

# Main content
tab1, tab2, tab3 = st.tabs(["Prediction", "Data Analysis", "Model Info"])

with tab1:
    if st.sidebar.button("Predict"):
        input_df = pd.DataFrame([inputs])
        
        # Prediction
        proba = model.predict_proba(input_df)[0][1]
        prediction = 1 if proba >= 0.5 else 0

        prediction_class = "Malignant" if proba >= 0.5 else "Benign"
        confidence = proba*100 if prediction_class == "Malignant" else (1-proba)*100

        # Color-coded display
        if prediction_class == "Malignant":
            st.error(f"Prediction: {prediction_class}")
        else:
            st.success(f"Prediction: {prediction_class}")

        # label = "Benign" if proba < 1 else "Malignant"
        # st.success(f"Prediction: {label}")
        st.info(f"Confidence: {confidence:.2f}%")
        
        # Pie Chart with explicit figure
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

        # SHAP Force Plot for SHAP >= v0.20
        st.subheader("Feature Impact Analysis")

        # 1. Initialize explainer
        explainer = shap.TreeExplainer(model)

        # 2. Prepare input data (ensure 2D array)
        input_array = input_df.values.reshape(1, -1)

        # 3. Get SHAP values
        shap_values = explainer(input_array)  # Returns Explanation object

        # 4. Create visualization
        plt.figure(figsize=(10, 3))
        shap.plots.force(
            base_value=explainer.expected_value[0],  # Use [0] for binary classification
            shap_values=shap_values.values[0, :, 0],  # For first (only) output
            features=input_array[0],
            feature_names=X.columns.tolist(),
            matplotlib=True,
            show=False
        )

        # 5. Display in Streamlit
        st.pyplot(plt.gcf())

with tab2:
    # Data Distribution Plots
    st.subheader("Data Distribution")
    
    # Create string version of diagnosis for plotting
    data['diagnosis_label'] = data['diagnosis'].map({0: 'Benign', 1: 'Malignant'})
    
    # Diagnosis Distribution Pie Chart with explicit figure
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
    
    # Feature Distribution Boxplot with explicit figure
    st.subheader("Feature Distribution by Diagnosis")
    selected_feature = st.selectbox("Select feature to visualize", X.columns)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        x='diagnosis_label',
        y=selected_feature,
        data=data,
        palette={'Benign': '#66b3ff', 'Malignant': '#ff9999'},
        width=0.5,
        ax=ax3
    )
    ax3.set_title(f'Distribution of {selected_feature} by Diagnosis')
    ax3.set_xlabel('Diagnosis')
    ax3.set_ylabel(selected_feature)
    st.pyplot(fig3)

with tab3:
    # Model Information
    st.subheader("Model Performance Metrics")
    
    # Feature Importance Plot with explicit figure
    st.subheader("Feature Importance")
    importances = model.feature_importances_
    indices = np.argsort(importances)[-10:]
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.barh(range(len(indices)), importances[indices], color='b', align='center')
    ax4.set_yticks(range(len(indices)))
    ax4.set_yticklabels(X.columns[indices])
    ax4.set_xlabel('Relative Importance')
    ax4.set_title('Top 10 Important Features')
    st.pyplot(fig4)
    
    # Confusion Matrix with explicit figure
    st.subheader("Confusion Matrix")
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig5, ax5 = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=['Benign', 'Malignant']).plot(ax=ax5)
    st.pyplot(fig5)

# Show raw data option
if st.checkbox("Show raw data"):
    st.write(data)







