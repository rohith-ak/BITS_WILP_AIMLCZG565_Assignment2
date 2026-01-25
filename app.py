# app.py
# Streamlit application for Adult Census Income Classification

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import sys
import joblib
import base64

# ------------------------------------------------------------------------------
# Add project root to path
# ------------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from model.src.feature_engineering.data import load_dataset
from model.src.visualization import (
    plot_confusion_matrix,
    plot_feature_importance,
)
from model.src.metrics_generation.metrics import calculate_metrics

# ------------------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Adult Census Income Prediction",
    layout="wide",
)

# ------------------------------------------------------------------------------
# Title and description
# ------------------------------------------------------------------------------
st.title("Adult Census Income Classification")
st.markdown("---")
st.write(
    """
This application allows you to:
- Download sample test dataset  
- Upload your test CSV dataset  
- Select from 6 pre-trained ML models  
- Get predictions and comprehensive evaluation metrics  
- Visualize confusion matrix and classification report  
"""
)

# ------------------------------------------------------------------------------
# Define saved models directory and model mapping
# ------------------------------------------------------------------------------
SAVED_MODELS_DIR = project_root / "model" / "saved_models"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbor": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
}

# ------------------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------------------
def create_download_link(df: pd.DataFrame, filename: str) -> str:
    """Create a download link for a dataframe."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = (
        f'<a href="data:file/csv;base64,{b64}" download="{filename}">'
        f"📥 Download {filename}</a>"
    )
    return href


def load_trained_model(model_name: str):
    """Load a trained model from ./model/saved_models."""
    model_path = SAVED_MODELS_DIR / MODEL_FILES[model_name]
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Please train the model first."
        )
    return joblib.load(model_path)


def display_metrics(metrics: dict):
    """Display metrics in a nice column layout."""
    st.subheader("Model Performance Metrics")

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


# ------------------------------------------------------------------------------
# Sidebar configuration
# ------------------------------------------------------------------------------
st.sidebar.header("Configuration")

# ------------------------------------------------------------------------------
# Download test dataset
# ------------------------------------------------------------------------------
st.sidebar.markdown("###Download Test Dataset")
test_data_path = project_root / "model" / "data" / "adult_test.csv"

if test_data_path.exists():
    test_df = pd.read_csv(test_data_path)
    st.sidebar.markdown(
        create_download_link(test_df, "adult_test.csv"),
        unsafe_allow_html=True,
    )
    # st.sidebar.info(
    #     f"Test dataset: {test_df.shape[0]} rows, {test_df.shape[1]} columns"
    # )
else:
    st.sidebar.warning("Test dataset not found!")

st.sidebar.markdown("---")

# ------------------------------------------------------------------------------
# File upload
# ------------------------------------------------------------------------------
uploaded_file = st.sidebar.file_uploader(
    "📤 Upload Test CSV Dataset",
    type=["csv"],
    help="Upload your test dataset in CSV format",
)

# ------------------------------------------------------------------------------
# Model selection
# ------------------------------------------------------------------------------
model_name = st.sidebar.selectbox(
    "Select Pre-trained Model",
    options=list(MODEL_FILES.keys()),
    help="Choose the pre-trained machine learning model",
)

model_path = SAVED_MODELS_DIR / MODEL_FILES[model_name]

if model_path.exists():
    st.sidebar.success(f"Model found: {MODEL_FILES[model_name]}")
else:
    st.sidebar.error("Model not found! Please train models first.")
    st.sidebar.info("Run: `python train_and_save_models.py`")

# ------------------------------------------------------------------------------
# Main content area
# ------------------------------------------------------------------------------
if uploaded_file is not None:
    try:
        # Save uploaded file temporarily
        temp_path = Path("temp_test_upload.csv")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success("Test dataset loaded successfully!")

        # Dataset preview
        df_preview = pd.read_csv(temp_path)
        with st.expander("View Test Dataset Preview"):
            st.dataframe(df_preview.head(10))

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Dataset Info:**")
            st.write(f"- Rows: {df_preview.shape[0]}")
            st.write(f"- Columns: {df_preview.shape[1]}")

        with col2:
            st.write("**Column Names:**")
            st.write(df_preview.columns.tolist())

        # ----------------------------------------------------------------------
        # Predict button
        # ----------------------------------------------------------------------
        st.sidebar.markdown("---")
        if st.sidebar.button("Predict & Evaluate", type="primary"):
            if not model_path.exists():
                st.error("❌ Model not found! Please train the model first.")
                st.info("Run: `python train_and_save_models.py`")
            else:
                with st.spinner(
                    f"Loading {model_name} and making predictions..."
                ):
                    try:
                        # Load and preprocess test data
                        X_test, y_test = load_dataset(
                            temp_path, use_feature_engineering=True
                        )

                        st.info(
                            f"Test Features shape: {X_test.shape}, "
                            f"Target shape: {y_test.shape}"
                        )

                        # Load model
                        model = load_trained_model(model_name)
                        st.success(
                            f"{model_name} loaded successfully from "
                            f"./model/saved_models"
                        )

                        # Predictions
                        y_pred = model.predict(X_test)

                        # Probabilities (if available)
                        y_prob = None
                        if hasattr(model, "predict_proba"):
                            y_prob = model.predict_proba(X_test)[:, 1]

                        # Metrics
                        metrics = calculate_metrics(
                            y_test, y_pred, y_prob
                        )

                        # Display metrics
                        st.markdown("---")
                        display_metrics(metrics)

                        # ------------------------------------------------------
                        # Confusion Matrix
                        # ------------------------------------------------------
                        st.markdown("---")
                        st.subheader("🧮 Confusion Matrix")

                        cm = confusion_matrix(y_test, y_pred)
                        class_names = ["<=50K", ">50K"]

                        fig = plot_confusion_matrix(
                            cm, class_names
                        )
                        st.plotly_chart(
                            fig, use_container_width=True
                        )

                        # ------------------------------------------------------
                        # Classification Report
                        # ------------------------------------------------------
                        st.markdown("---")
                        st.subheader("Classification Report")

                        report = classification_report(
                            y_test,
                            y_pred,
                            target_names=class_names,
                            output_dict=True,
                        )

                        report_df = (
                            pd.DataFrame(report).transpose()
                        )
                        st.dataframe(
                            report_df.style.format("{:.4f}"),
                            use_container_width=True,
                        )

                        # ------------------------------------------------------
                        # Feature importance (if available)
                        # ------------------------------------------------------
                        if hasattr(
                            model, "feature_importances_"
                        ):
                            st.markdown("---")
                            st.subheader("Feature Importance")

                            fig = plot_feature_importance(
                                model.feature_importances_,
                                X_test.columns,
                            )
                            st.plotly_chart(
                                fig, use_container_width=True
                            )

                        # ------------------------------------------------------
                        # Download predictions
                        # ------------------------------------------------------
                        st.markdown("---")
                        st.subheader("Download Predictions")

                        predictions_df = df_preview.copy()
                        predictions_df["predicted_income"] = [
                            "<=50K" if p == 0 else ">50K"
                            for p in y_pred
                        ]
                        predictions_df["actual_income"] = [
                            "<=50K" if p == 0 else ">50K"
                            for p in y_test
                        ]
                        predictions_df["correct"] = (
                            y_pred == y_test
                        )

                        if y_prob is not None:
                            predictions_df[
                                "probability_>50K"
                            ] = y_prob

                        st.markdown(
                            create_download_link(
                                predictions_df,
                                "predictions_with_results.csv",
                            ),
                            unsafe_allow_html=True,
                        )

                    except Exception as e:
                        st.error(
                            f"❌ Error during prediction: {str(e)}"
                        )
                        st.exception(e)

        # Cleanup temp file
        if temp_path.exists():
            temp_path.unlink()

    except Exception as e:
        st.error(
            f"Error loading test dataset: {str(e)}"
        )
        st.exception(e)

else:
    # Instructions when no file is uploaded
    st.info(
        "Please download the test dataset or upload your own CSV file to get started"
    )

    st.markdown("###Instructions:")
    st.markdown(
        """
1. **Download Test Dataset**: Click the download link in the sidebar to get sample test data  
2. **Upload Test Dataset**: Upload the downloaded CSV or your own test dataset  
3. **Select Model**: Choose from 6 pre-trained machine learning models  
4. **Predict & Evaluate**: Click the button to get predictions and evaluation metrics  
5. **View Results**: See confusion matrix, classification report, and metrics  
6. **Download Results**: Download predictions with actual vs predicted values  
"""
    )

    st.markdown("###Available Pre-trained Models:")
    for i, model in enumerate(MODEL_FILES.keys(), 1):
        status = (
            "✅"
            if (SAVED_MODELS_DIR / MODEL_FILES[model]).exists()
            else "❌"
        )
        st.markdown(f"{i}. {status} {model}")

    if not any(
        (SAVED_MODELS_DIR / f).exists()
        for f in MODEL_FILES.values()
    ):
        st.warning("No trained models found!")
        st.info(
            "To train models, run: `python train_and_save_models.py`"
        )

    st.markdown("###Metrics Displayed:")
    st.markdown(
        """
- **Accuracy**: Overall correctness of predictions  
- **AUC**: Area Under the ROC Curve  
- **Precision**: Proportion of positive predictions that are correct  
- **Recall**: Proportion of actual positives that are identified  
- **F1 Score**: Harmonic mean of precision and recall  
- **MCC**: Matthews Correlation Coefficient  
"""
    )

    st.markdown("### ✨ Features:")
    st.markdown(
        """
-  Pre-trained models with advanced feature engineering  
-  Download sample test dataset  
-  Upload custom test data  
-  Comprehensive evaluation metrics  
-  Visual confusion matrix  
-  Detailed classification report  
-  Feature importance (for tree-based models)  
-  Download predictions with results  
"""
    )

# ------------------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    '<div style="text-align: center;">'
    "Built with Streamlit | Adult Census Income Classification"
    "</div>",
    unsafe_allow_html=True,
)
