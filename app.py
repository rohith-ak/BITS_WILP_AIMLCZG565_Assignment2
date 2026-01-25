# app.py
# Optimized Streamlit application for Adult Census Income Classification

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import sys
import joblib
import base64
import hashlib
 
# ------------------------------------------------------------------------------
# Page configuration - MUST BE FIRST
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Adult Census Income Prediction",
    layout="wide",
)

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
# Initialize session state BEFORE any other operations
# ------------------------------------------------------------------------------
def init_session_state():
    """Initialize all session state variables."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.models_loaded = False
        st.session_state.models = {}
        st.session_state.processed_data = None
        st.session_state.file_hash = None
        st.session_state.raw_data = None
        st.session_state.X_test = None
        st.session_state.y_test = None

# Call initialization
init_session_state()

# ------------------------------------------------------------------------------
# Cached functions for performance
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading all models... (one-time operation)")
def load_all_models():
    """Load all models at once and cache them."""
    models = {}
    for model_name, model_file in MODEL_FILES.items():
        model_path = SAVED_MODELS_DIR / model_file
        if model_path.exists():
            try:
                models[model_name] = joblib.load(model_path)
            except Exception as e:
                st.warning(f"Failed to load {model_name}: {str(e)}")
    return models

@st.cache_data(show_spinner="Loading test dataset template...")
def load_test_dataset_template():
    """Cache the test dataset for download."""
    test_data_path = project_root / "model" / "data" / "adult_test.csv"
    if test_data_path.exists():
        return pd.read_csv(test_data_path)
    return None

# ------------------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------------------
def get_file_hash(uploaded_file) -> str:
    """Generate hash of uploaded file to detect changes."""
    uploaded_file.seek(0)
    file_hash = hashlib.md5(uploaded_file.read()).hexdigest()
    uploaded_file.seek(0)
    return file_hash

def create_download_link(df: pd.DataFrame, filename: str) -> str:
    """Create a download link for a dataframe."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = (
        f'<a href="data:file/csv;base64,{b64}" download="{filename}">'
        f"📥 Download {filename}</a>"
    )
    return href

def process_uploaded_data(uploaded_file):
    """Process uploaded file and cache in session state."""
    try:
        file_hash = get_file_hash(uploaded_file)
        
        # Check if we've already processed this exact file
        if (st.session_state.file_hash == file_hash and 
            st.session_state.X_test is not None and
            st.session_state.y_test is not None):
            return st.session_state.X_test, st.session_state.y_test
        
        # Process new file
        temp_path = Path("temp_test_upload.csv")
        uploaded_file.seek(0)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Load raw data for preview
        raw_df = pd.read_csv(temp_path)
        
        # Load and preprocess
        X_test, y_test = load_dataset(temp_path, use_feature_engineering=True)
        
        # Clean up
        if temp_path.exists():
            temp_path.unlink()
        
        # Cache in session state
        st.session_state.file_hash = file_hash
        st.session_state.raw_data = raw_df
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        
        return X_test, y_test
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        raise

def display_metrics(metrics: dict, model_name: str):
    """Display metrics in a nice column layout."""
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

# ------------------------------------------------------------------------------
# Title and description
# ------------------------------------------------------------------------------
st.title("Adult Census Income Classification")
with st.expander("Application Overview"):
    st.write(
        """
    This application allows you to:
    - Download sample test dataset  
    - Upload your test CSV dataset (cached for fast re-use)
    - Select from 6 pre-trained ML models (all pre-loaded)
    - Get predictions and comprehensive evaluation metrics  
    - Visualize confusion matrix and classification report  
    
    **Performance Features:**
    - All models pre-loaded in memory
    - Uploaded data cached - no re-processing when switching models
    - Instant predictions after first upload
    """
    )

# ------------------------------------------------------------------------------
# Pre-load all models (happens once)
# ------------------------------------------------------------------------------
if not st.session_state.models_loaded:
    st.session_state.models = load_all_models()
    st.session_state.models_loaded = True

# ------------------------------------------------------------------------------
# Sidebar configuration
# ------------------------------------------------------------------------------
st.sidebar.header("Configuration")

# Show model loading status
models_count = len(st.session_state.models)
if models_count > 0:
    st.sidebar.success(f"✅ {models_count} models pre-loaded")
else:
    st.sidebar.error("⚠️ No models loaded!")
    st.sidebar.info("Run: `python train_and_save_models.py`")

# ------------------------------------------------------------------------------
# Download test dataset
# ------------------------------------------------------------------------------
st.sidebar.markdown("### Download Test Dataset")
test_df_template = load_test_dataset_template()

if test_df_template is not None:
    st.sidebar.markdown(
        create_download_link(test_df_template, "adult_test.csv"),
        unsafe_allow_html=True,
    )
else:
    st.sidebar.warning("Test dataset not found!")

# ------------------------------------------------------------------------------
# File upload
# ------------------------------------------------------------------------------
uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV Dataset",
    type=["csv"],
    help="Upload your test dataset in CSV format (will be cached)",
)

# ------------------------------------------------------------------------------
# Model selection
# ------------------------------------------------------------------------------
available_models = list(st.session_state.models.keys())
if available_models:
    model_name = st.sidebar.selectbox(
        "Select Pre-trained Model",
        options=available_models,
        help="Choose the pre-trained machine learning model (already loaded)",
    )
else:
    model_name = st.sidebar.selectbox(
        "Select Pre-trained Model",
        options=list(MODEL_FILES.keys()),
        help="No models loaded yet",
        disabled=True,
    )

# ------------------------------------------------------------------------------
# Main content area
# ------------------------------------------------------------------------------
if uploaded_file is not None:
    try:
        # Process uploaded file (cached if same file)
        X_test, y_test = process_uploaded_data(uploaded_file)
        
        st.success("✅ Test dataset loaded and processed (cached in memory)")

        # Dataset preview
        df_preview = st.session_state.raw_data
        
        if df_preview is not None:
            with st.expander("View Test Dataset Preview"):
                st.dataframe(df_preview.head(10))

            with st.expander("Dataset and Column Info"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Dataset Info:**")
                    st.write(f"- Rows: {df_preview.shape[0]}")
                    st.write(f"- Columns: {df_preview.shape[1]}")

                with col2:
                    st.write("**Column Names:**")
                    st.write(df_preview.columns.tolist())

        # Display processed data info
        st.info(
            f"✅ Preprocessed Features: {X_test.shape[0]} samples, "
            f"{X_test.shape[1]} features | Target: {y_test.shape[0]} samples"
        )

        # ----------------------------------------------------------------------
        # Predict button
        # ----------------------------------------------------------------------
        if st.sidebar.button("🚀 Predict & Evaluate", type="primary"):
            if model_name not in st.session_state.models:
                st.error("❌ Selected model not loaded!")
            else:
                try:
                    # Get model from cache
                    model = st.session_state.models[model_name]

                    # Predictions (fast - data already processed)
                    y_pred = model.predict(X_test)

                    # Probabilities (if available)
                    y_prob = None
                    if hasattr(model, "predict_proba"):
                        y_prob = model.predict_proba(X_test)[:, 1]

                    # Metrics
                    metrics = calculate_metrics(y_test, y_pred, y_prob)

                    # Display metrics
                    st.markdown("---")
                    display_metrics(metrics, model_name)

                    # ------------------------------------------------------
                    # Confusion Matrix
                    # ------------------------------------------------------
                    st.markdown("---")
                    st.subheader("🧮 Confusion Matrix")

                    cm = confusion_matrix(y_test, y_pred)
                    class_names = ["<=50K", ">50K"]

                    fig = plot_confusion_matrix(cm, class_names)
                    st.plotly_chart(fig, use_container_width=True)

                    # ------------------------------------------------------
                    # Classification Report
                    # ------------------------------------------------------
                    st.markdown("---")
                    st.subheader("📊 Classification Report")

                    report = classification_report( 
                        y_test,
                        y_pred,
                        target_names=class_names,
                        output_dict=True,
                    )

                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(
                        report_df.style.format("{:.4f}"),
                        use_container_width=True,
                    )

                    # ------------------------------------------------------
                    # Feature importance (if available)
                    # ------------------------------------------------------
                    if hasattr(model, "feature_importances_"):
                        st.markdown("---")
                        st.subheader("📈 Feature Importance")

                        fig = plot_feature_importance(
                            model.feature_importances_,
                            X_test.columns,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # ------------------------------------------------------
                    # Download predictions
                    # ------------------------------------------------------
                    st.markdown("---")
                    st.subheader("💾 Download Predictions")

                    predictions_df = df_preview.copy()
                    predictions_df["predicted_income"] = [
                        "<=50K" if p == 0 else ">50K" for p in y_pred
                    ]
                    predictions_df["actual_income"] = [
                        "<=50K" if p == 0 else ">50K" for p in y_test
                    ]
                    predictions_df["correct"] = y_pred == y_test

                    if y_prob is not None:
                        predictions_df["probability_>50K"] = y_prob

                    st.markdown(
                        create_download_link(
                            predictions_df,
                            f"predictions_{model_name.lower().replace(' ', '_')}.csv",
                        ),
                        unsafe_allow_html=True,
                    )

                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
                    st.exception(e)

    except Exception as e:
        st.error(f"Error processing dataset: {str(e)}")
        st.exception(e)

else:
    # Instructions when no file is uploaded
    st.info("👆 Please download the test dataset or upload your own CSV file to get started")

    st.markdown("### 📋 Instructions:")
    st.markdown(
        """
1. **Download Test Dataset**: Click the download link in the sidebar to get sample test data  
2. **Upload Test Dataset**: Upload the downloaded CSV or your own test dataset  
3. **Select Model**: Choose from pre-loaded machine learning models  
4. **Predict & Evaluate**: Click the button to get instant predictions  
5. **Switch Models**: Change model and predict again - data stays cached!  
6. **Download Results**: Download predictions with actual vs predicted values  
"""
    )

    st.markdown("### 🤖 Available Pre-trained Models:")
    for i, model in enumerate(MODEL_FILES.keys(), 1):
        status = "✅" if model in st.session_state.models else "❌"
        st.markdown(f"{i}. {status} {model}")

    if not st.session_state.models:
        st.warning("⚠️ No trained models found!")
        st.info("To train models, run: `python train_and_save_models.py`")

    st.markdown("### 📊 Metrics Displayed:")
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

    st.markdown("### ⚡ Performance Features:")
    st.markdown(
        """
- 🚀 **All models pre-loaded** - No loading time when switching models  
- 💾 **Data caching** - Upload once, test multiple models instantly  
- 🔄 **Smart re-processing** - Only processes data when file changes  
- ⚡ **Instant predictions** - Sub-second response after first upload  
"""
    )

# ------------------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    '<div style="text-align: center;">'
    "Built with Streamlit | Adult Census Income Classification | ⚡ Optimized for Speed"
    "</div>",
    unsafe_allow_html=True,
)