# import streamlit as st
# import pandas as pd
# import numpy as np
# import shap
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# # 1. Load and prepare data
# @st.cache_data
# def load_data():
#     url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
#     columns = ["id", "diagnosis"] + [f"feature_{i}" for i in range(1, 31)]
#     data = pd.read_csv(url, header=None, names=columns)
#     data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})
#     return data

# data = load_data()
# X = data.drop(['id', 'diagnosis'], axis=1)
# y = data['diagnosis']

# # 2. Train/test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, 
#     test_size=0.2, 
#     random_state=42,
#     stratify=y
# )

# # 3. Train model
# @st.cache_resource
# def train_model():
#     model = RandomForestClassifier(
#         n_estimators=100,
#         random_state=42,
#         class_weight='balanced'
#     )
#     model.fit(X_train, y_train)
#     return model

# model = train_model()

# # 4. Streamlit UI
# st.title("Breast Cancer Prediction System")

# pred = model.predict(X)[0]

# # Sidebar inputs
# st.sidebar.header("Patient Features")
# inputs = {}
# for feature in X.columns:
#     inputs[feature] = st.sidebar.slider(
#         feature,
#         float(X[feature].min()),
#         float(X[feature].max()),
#         float(X[feature].mean())
#     )

# # Main content
# tab1, tab2, tab3 = st.tabs(["Prediction", "Data Analysis", "Model Info"])

# with tab1:
#     if st.sidebar.button("Predict"):
#         input_df = pd.DataFrame([inputs])
        
#         # Prediction
#         proba = model.predict_proba(input_df)[0][1]
#         prediction = 1 if proba >= 0.5 else 0

#         prediction_class = "Malignant" if proba >= 0.5 else "Benign"
#         confidence = proba*100 if prediction_class == "Malignant" else (1-proba)*100

#         # Color-coded display
#         if prediction_class == "Malignant":
#             st.error(f"Prediction: {prediction_class}")
#         else:
#             st.success(f"Prediction: {prediction_class}")

#         # label = "Benign" if proba < 1 else "Malignant"
#         # st.success(f"Prediction: {label}")
#         st.info(f"Confidence: {confidence:.2f}%")
        
#         # Pie Chart with explicit figure
#         fig1, ax1 = plt.subplots()
#         ax1.pie(
#             [proba, 1-proba],
#             labels=['Malignant', 'Benign'],
#             colors=['#ff9999','#66b3ff'],
#             autopct='%1.1f%%',
#             startangle=90
#         )
#         ax1.axis('equal')
#         st.subheader("Prediction Probability")
#         st.pyplot(fig1)

#         # SHAP Force Plot for SHAP >= v0.20
#         st.subheader("Feature Impact Analysis")

#         # 1. Initialize explainer
#         explainer = shap.TreeExplainer(model)

#         # 2. Prepare input data (ensure 2D array)
#         input_array = input_df.values.reshape(1, -1)

#         # 3. Get SHAP values
#         shap_values = explainer(input_array)  # Returns Explanation object

#         # 4. Create visualization
#         plt.figure(figsize=(10, 3))
#         shap.plots.force(
#             base_value=explainer.expected_value[0],  # Use [0] for binary classification
#             shap_values=shap_values.values[0, :, 0],  # For first (only) output
#             features=input_array[0],
#             feature_names=X.columns.tolist(),
#             matplotlib=True,
#             show=False
#         )

#         # 5. Display in Streamlit
#         st.pyplot(plt.gcf())

# with tab2:
#     # Data Distribution Plots
#     st.subheader("Data Distribution")
    
#     # Create string version of diagnosis for plotting
#     data['diagnosis_label'] = data['diagnosis'].map({0: 'Benign', 1: 'Malignant'})
    
#     # Diagnosis Distribution Pie Chart with explicit figure
#     fig2, ax2 = plt.subplots()
#     diagnosis_counts = data['diagnosis_label'].value_counts()
#     ax2.pie(
#         diagnosis_counts,
#         labels=diagnosis_counts.index,
#         colors=['#66b3ff','#ff9999'],
#         autopct='%1.1f%%',
#         startangle=90
#     )
#     ax2.axis('equal')
#     st.pyplot(fig2)
    
#     # Feature Distribution Boxplot with explicit figure
#     st.subheader("Feature Distribution by Diagnosis")
#     selected_feature = st.selectbox("Select feature to visualize", X.columns)
#     fig3, ax3 = plt.subplots(figsize=(10, 6))
#     sns.boxplot(
#         x='diagnosis_label',
#         y=selected_feature,
#         data=data,
#         palette={'Benign': '#66b3ff', 'Malignant': '#ff9999'},
#         width=0.5,
#         ax=ax3
#     )
#     ax3.set_title(f'Distribution of {selected_feature} by Diagnosis')
#     ax3.set_xlabel('Diagnosis')
#     ax3.set_ylabel(selected_feature)
#     st.pyplot(fig3)

# with tab3:
#     # Model Information
#     st.subheader("Model Performance Metrics")
    
#     # Feature Importance Plot with explicit figure
#     st.subheader("Feature Importance")
#     importances = model.feature_importances_
#     indices = np.argsort(importances)[-10:]
#     fig4, ax4 = plt.subplots(figsize=(10, 6))
#     ax4.barh(range(len(indices)), importances[indices], color='b', align='center')
#     ax4.set_yticks(range(len(indices)))
#     ax4.set_yticklabels(X.columns[indices])
#     ax4.set_xlabel('Relative Importance')
#     ax4.set_title('Top 10 Important Features')
#     st.pyplot(fig4)
    
#     # Confusion Matrix with explicit figure
#     st.subheader("Confusion Matrix")
#     y_pred = model.predict(X_test)
#     cm = confusion_matrix(y_test, y_pred)
#     fig5, ax5 = plt.subplots()
#     ConfusionMatrixDisplay(cm, display_labels=['Benign', 'Malignant']).plot(ax=ax5)
#     st.pyplot(fig5)

# # Show raw data option
# if st.checkbox("Show raw data"):
#     st.write(data)







import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Actual feature names from the Wisconsin Breast Cancer dataset
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 
    'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se', 
    'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
    'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# 1. Load and prepare data
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    columns = ["id", "diagnosis"] + feature_names
    data = pd.read_csv(url, header=None, names=columns)
    data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})
    return data

data = load_data()
X = data.drop(['id', 'diagnosis'], axis=1)
y = data['diagnosis']

# Save original feature order
feature_order = X.columns.tolist()

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

# Get top 10 important features
importances = model.feature_importances_
top_10_indices = np.argsort(importances)[-10:]
top_10_features = X.columns[top_10_indices]

# 4. Streamlit UI
st.title("Breast Cancer Prediction System")

# Sidebar with top 10 features
st.sidebar.header("Patient Features")
# st.sidebar.markdown("**Feature Importance Rankings:**")
# for i, (feature, imp) in enumerate(zip(top_10_features, sorted(importances[top_10_indices])[::-1])):
#     st.sidebar.markdown(f"{i+1}. {feature.replace('_', ' ').title()} ({imp:.3f})")

inputs = {}
for feature in top_10_features:
    # Create human-readable labels
    readable_name = feature.replace('_', ' ').title()
    inputs[feature] = st.sidebar.slider(
        readable_name,
        float(X[feature].min()),
        float(X[feature].max()),
        float(X[feature].mean()),
        step=0.01
    )

# Fill remaining features with mean values
for feature in feature_order:
    if feature not in inputs:
        inputs[feature] = float(X[feature].mean())

# Main content
tab1, tab2, tab3 = st.tabs(["Prediction", "Data Analysis", "Model Info"])

with tab1:
    if st.sidebar.button("Predict", type="primary"):
        # Create DataFrame with features in original training order
        input_data = {feature: [inputs[feature]] for feature in feature_order}
        input_df = pd.DataFrame(input_data, columns=feature_order)
        
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

        # SHAP Force Plot
        st.subheader("Feature Impact Analysis")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(input_df.values.reshape(1, -1))
        
        plt.figure(figsize=(10, 3))
        shap.plots.force(
            base_value=explainer.expected_value[0],
            shap_values=shap_values.values[0, :, 0],
            features=input_df.iloc[0],
            feature_names=[name.replace('_', ' ').title() for name in feature_order],
            matplotlib=True,
            show=False
        )
        st.pyplot(plt.gcf())

with tab2:
    # Data Distribution Plots
    st.subheader("Data Distribution")
    
    data['diagnosis_label'] = data['diagnosis'].map({0: 'Benign', 1: 'Malignant'})
    
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
    # Use human-readable names for the selectbox
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
    # Model Information
    st.subheader("Model Performance Metrics")
    
    # Feature Importance Plot with readable names
    st.subheader("Top 10 Important Features")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    readable_names = [name.replace('_', ' ').title() for name in top_10_features]
    ax4.barh(range(len(top_10_features)), importances[top_10_indices], color='#4e73df', align='center')
    ax4.set_yticks(range(len(top_10_features)))
    ax4.set_yticklabels(readable_names)
    ax4.set_xlabel('Relative Importance')
    st.pyplot(fig4)
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig5, ax5 = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=['Benign', 'Malignant']).plot(ax=ax5)
    st.pyplot(fig5)

# Show raw data option
if st.checkbox("Show raw data"):
    st.write(data)