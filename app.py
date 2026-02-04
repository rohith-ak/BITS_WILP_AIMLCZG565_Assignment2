"""
This Streamlit application provides an interactive web interface for predicting income levels (<=50K or >50K) 
from the Adult Census dataset using six pre-trained machine learning models 
including Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, and XGBoost. 
The app allows users to download a sample test dataset or upload their own CSV file, automatically performs feature engineering 
on the uploaded data, and generates comprehensive evaluation metrics 
including accuracy, precision, recall, F1 score, MCC, and AUC along with visual representations like confusion matrices and feature importance plots. 
Users can view predictions, download results with actual vs predicted values, and compare performance across different models through an 
intuitive sidebar-based navigation system.
"""


import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import sys
import joblib
import base64

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from model.src.feature_engineering.data import load_dataset
from model.src.visualization import plot_confusion_matrix, plot_feature_importance
from model.src.metrics_generation.metrics import calculate_metrics

# Page configuration
st.set_page_config(
    page_title="Adult Census Income Prediction",
    layout="wide"
)

# Title and description
st.title("Adult Census Income Classification")
with st.expander("Application Information"):
    st.write("""
    This application allows you to:
    - Download sample test dataset
    - Upload your test csv dataset
    - Select from 6 pre-trained ML models
    - Get predictions and comprehensive evaluation metrics
    - Visualize confusion matrix and classification report
    """)

# Define saved_models directory at project root
SAVED_MODELS_DIR = project_root / "model/saved_models"
TEMP_ENGINEERED_DATA_PATH = project_root / "model/temp_engineered_data.csv"

# Model files mapping
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbor": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
}

def create_download_link(df, filename):
    """Create a download link for a dataframe."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def load_trained_model(model_name):
    """Load a trained model from ./saved_models directory."""
    model_path = SAVED_MODELS_DIR / MODEL_FILES[model_name]
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    return joblib.load(model_path)

def perform_feature_engineering(input_csv_path):
    """Perform feature engineering and save to temporary path."""
    try:
        X, y = load_dataset(input_csv_path, use_feature_engineering=True)
        engineered_df = pd.DataFrame(X)
        engineered_df["target"] = y
        engineered_df.to_csv(TEMP_ENGINEERED_DATA_PATH, index=False)
        return X, y, engineered_df
    except Exception as e:
        raise Exception(f"Feature engineering failed: {str(e)}")

def display_metrics(metrics, model_name):
    """Display metrics in a nice format with columns."""
    st.subheader(f"Model Performance Metrics - {model_name}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        st.metric("Precision", f"{metrics['precision']:.4f}")

    with col2:
        st.metric("Recall", f"{metrics['recall']:.4f}")
        st.metric("F1 Score", f"{metrics['f1_score']:.4f}")

    with col3:
        st.metric("MCC", f"{metrics['mcc']:.4f}")
        if "auc" in metrics:
            st.metric("AUC", f"{metrics['auc']:.4f}")

# Sidebar for inputs
st.sidebar.header("Configuration")

# Download Test Dataset Section
st.sidebar.markdown("### Download Test Dataset")
test_data_path = project_root / "model/data" / "adult_test.csv"

if test_data_path.exists():
    with st.spinner("Loading test dataset..."):
        test_df = pd.read_csv(test_data_path)
    st.sidebar.markdown(create_download_link(test_df, "adult_test.csv"), unsafe_allow_html=True)
    st.sidebar.info(f"Test dataset: {test_df.shape[0]} rows, {test_df.shape[1]} columns")
else:
    st.sidebar.warning("Test dataset not found!")


uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV Dataset",
    type=["csv"],
    help="Upload your test dataset in CSV format"
)

if "engineered_data_ready" not in st.session_state:
    st.session_state.engineered_data_ready = False
if "x_test" not in st.session_state:
    st.session_state.x_test = None
if "y_test" not in st.session_state:
    st.session_state.y_test = None
if "original_df" not in st.session_state:
    st.session_state.original_df = None
if "show_results" not in st.session_state:
    st.session_state.show_results = False
if "results_data" not in st.session_state:
    st.session_state.results_data = None

if uploaded_file is not None:
    try:
        temp_upload_path = Path("temp_upload.csv")
        with open(temp_upload_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Loading uploaded dataset..."):
            df_preview = pd.read_csv(temp_upload_path)
            st.session_state.original_df = df_preview

        st.success("Test dataset loaded successfully!")

        with st.spinner("Performing feature engineering..."):
            X_test, y_test, engineered_df = perform_feature_engineering(temp_upload_path)
            st.session_state.x_test = X_test
            st.session_state.y_test = y_test
            st.session_state.engineered_data_ready = True

        st.success(f"Feature engineering completed! Engineered features: {X_test.shape[1]}")

        with st.expander("View Test Dataset Preview"):
            if st.button("Show Dataset Preview"):
                st.dataframe(df_preview.head(10))

                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Original Dataset Info:**")
                    st.write(f"- Rows: {df_preview.shape[0]}")
                    st.write(f"- Columns: {df_preview.shape[1]}")
                with col2:
                    st.write("**Engineered Dataset Info:**")
                    st.write(f"- Rows: {X_test.shape[0]}")
                    st.write(f"- Features: {X_test.shape[1]}")

        if temp_upload_path.exists():
            temp_upload_path.unlink()

    except Exception as e:
        st.error(f"Error processing test dataset: {str(e)}")
        st.exception(e)
        st.session_state.engineered_data_ready = False

if st.session_state.engineered_data_ready:
    model_name = st.sidebar.selectbox(
        "Select Pre-trained Model",
        options=list(MODEL_FILES.keys()),
        help="Choose the pre-trained machine learning model"
    )

    model_path = SAVED_MODELS_DIR / MODEL_FILES[model_name]

    if model_path.exists():
        st.sidebar.success(f"Model found: {MODEL_FILES[model_name]}")
    else:
        st.sidebar.error("Model not found! Please train models first.")
        st.sidebar.info("Run: python train_and_save_models.py")
 
    col1, col2 = st.sidebar.columns(2)

    with col1:
        predict_button = st.button("Predict & Evaluate", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("Clear Output", use_container_width=True)
    
    st.sidebar.info("Model loading and prediction may take a few moments. Please be patient.")

    if clear_button:
        st.session_state.show_results = False
        st.session_state.results_data = None
        st.rerun()

    if predict_button:
        st.session_state.show_results = False
        st.session_state.results_data = None

        if not model_path.exists():
            st.error("Model not found! Please train the model first.")
            st.info("Run: python train_and_save_models.py to train and save all models.")
        else:
            with st.spinner(f"Loading {model_name} and making predictions..."):
                try:
                    X_test = st.session_state.x_test
                    y_test = st.session_state.y_test
                    df_preview = st.session_state.original_df

                    with st.spinner(f"Loading {model_name} model..."):
                        model = load_trained_model(model_name)
                    st.success(f"{model_name} loaded successfully from ./model/saved_models")

                    with st.spinner("Making predictions..."):
                        y_pred = model.predict(X_test)

                    y_prob = None
                    if hasattr(model, "predict_proba"):
                        with st.spinner("Calculating probabilities..."):
                            y_prob = model.predict_proba(X_test)[:, 1]

                    with st.spinner("Calculating metrics..."):
                        metrics = calculate_metrics(y_test, y_pred, y_prob)

                    with st.spinner("Generating visualizations..."):
                        cm = confusion_matrix(y_test, y_pred)
                        class_names = ["<=50K", ">50K"]
                        cm_fig = plot_confusion_matrix(cm, class_names)

                    report = classification_report(
                        y_test,
                        y_pred,
                        target_names=class_names,
                        output_dict=True
                    )
                    report_df = pd.DataFrame(report).transpose()

                    fi_fig = None
                    if hasattr(model, "feature_importances_"):
                        fi_fig = plot_feature_importance(
                            model.feature_importances_,
                            X_test.columns
                        )

                    predictions_df = df_preview.copy()
                    predictions_df["predicted_income"] = ["<=50K" if p == 0 else ">50K" for p in y_pred]
                    predictions_df["actual_income"] = ["<=50K" if p == 0 else ">50K" for p in y_test]
                    predictions_df["correct"] = (y_pred == y_test)

                    if y_prob is not None:
                        predictions_df["probability_>50K"] = y_prob

                    st.session_state.results_data = {
                        "model_name": model_name,
                        "metrics": metrics,
                        "cm_fig": cm_fig,
                        "report_df": report_df,
                        "fi_fig": fi_fig,
                        "predictions_df": predictions_df,
                        "class_names": class_names,
                    }
                    st.session_state.show_results = True

                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    st.exception(e)

if st.session_state.show_results and st.session_state.results_data is not None:
    results = st.session_state.results_data

    st.markdown("---")
    display_metrics(results["metrics"], results["model_name"])

    st.markdown("---")
    st.subheader("Confusion Matrix")
    st.plotly_chart(results["cm_fig"], use_container_width=True)

    st.markdown("---")
    st.subheader("Classification Report")
    st.dataframe(
        results["report_df"].style.format("{:.4f}"),
        use_container_width=True,
    )

    if results["fi_fig"] is not None:
        st.markdown("---")
        st.subheader("Feature Importance")
        st.plotly_chart(results["fi_fig"], use_container_width=True)

    st.markdown("---")
    st.subheader("Download Predictions")
    st.markdown(
        create_download_link(results["predictions_df"], "predictions_with_results.csv"),
        unsafe_allow_html=True,
    )

elif not st.session_state.engineered_data_ready:
    st.info("Please download the test dataset or upload your own CSV file to get started")

    st.markdown("### Instructions:")
    st.markdown("""
    1. **Download Test Dataset**: Click the download link in the sidebar to get sample test data  
    2. **Upload Test Dataset**: Upload the downloaded CSV or your own test dataset  
    3. **Feature Engineering**: System will automatically perform feature engineering  
    4. **Select Model**: Choose from 6 pre-trained machine learning models  
    5. **Predict & Evaluate**: Click the button to get predictions and evaluation metrics  
    6. **View Results**: See confusion matrix, classification report, and metrics  
    7. **Download Results**: Download predictions with actual vs predicted values  
    """)

    st.markdown("### Available Pre-trained Models:")
    for i, model in enumerate(MODEL_FILES.keys(), 1):
        status = "Available" if (SAVED_MODELS_DIR / MODEL_FILES[model]).exists() else "Not Found"
        st.markdown(f"{i}. {model} - {status}")

    if not any((SAVED_MODELS_DIR / f).exists() for f in MODEL_FILES.values()):
        st.warning("No trained models found!")
        st.info("**To train models, run:** python train_and_save_models.py")

    st.markdown("### Metrics Displayed:")
    st.markdown("""
    - **Accuracy**: Overall correctness of predictions  
    - **AUC**: Area Under the ROC Curve  
    - **Precision**: Proportion of positive predictions that are correct  
    - **Recall**: Proportion of actual positives that are identified  
    - **F1 Score**: Harmonic mean of precision and recall  
    - **MCC**: Matthews Correlation Coefficient  
    """)

    st.markdown("### Features:")
    st.markdown("""
    - Pre-trained models with advanced feature engineering  
    - Download sample test dataset  
    - Upload custom test data  
    - Automatic feature engineering pipeline  
    - Comprehensive evaluation metrics  
    - Visual confusion matrix  
    - Detailed classification report  
    - Feature importance (for tree-based models)  
    - Download predictions with results  
    """)

st.markdown("---")
st.markdown(
    "<div style='text-align: center'>Built with Streamlit | Adult Census Income Classification</div>",
    unsafe_allow_html=True
)
 