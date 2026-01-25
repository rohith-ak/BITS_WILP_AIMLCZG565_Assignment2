"""
Script to train all models with feature engineering and save them to ./saved_models.
Run this script once to train and save all models.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import joblib
import sys

# Add project root to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent  # Go up two levels to project root
sys.path.insert(0, str(project_root))

from model.src.feature_engineering.data import load_dataset
from model.src.model_code.logistic_regression import LogisticRegressionModel
from model.src.model_code.decision_tree import DecisionTreeModel
from model.src.model_code.knn import KNNModel
from model.src.model_code.naive_bayes import NaiveBayesModel
from model.src.model_code.random_forest import RandomForestModel
from model.src.model_code.xgboost_model import XGBoostModel


def train_and_save_all_models():
    """Train all models with feature engineering and save them to ./saved_models."""
    print("=" * 80)
    print("TRAINING AND SAVING ALL MODELS")
    print("=" * 80)

    # Load training data (from project root)
    data_path = project_root / "model" / "data" / "adult_train.csv"

    if not data_path.exists():
        print(f"❌ Error: Training data not found at {data_path}")
        print(f"Looking at: {data_path.absolute()}")
        return

    print(f"\n📂 Loading training data from: {data_path}")

    # Load with feature engineering
    X, y = load_dataset(data_path, use_feature_engineering=True)

    print("✅ Data loaded successfully!")
    print(f" - Features shape: {X.shape}")
    print(f" - Target shape: {y.shape}")
    print(f" - Class distribution: {pd.Series(y).value_counts().to_dict()}")

    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n📊 Data split:")
    print(f" - Training samples: {X_train.shape[0]}")
    print(f" - Validation samples: {X_val.shape[0]}")

    # Create saved_models directory at project root/model/
    models_dir = project_root / "model" / "saved_models"
    models_dir.mkdir(exist_ok=True)
    print(f"\n💾 Models will be saved to: {models_dir}")

    # Define models
    models_config = {
        "Logistic Regression": (LogisticRegressionModel(), "logistic_regression.joblib"),
        "Decision Tree": (DecisionTreeModel(), "decision_tree.joblib"),
        "K-Nearest Neighbor": (KNNModel(), "knn.joblib"),
        "Naive Bayes": (NaiveBayesModel(), "naive_bayes.joblib"),
        "Random Forest": (RandomForestModel(), "random_forest.joblib"),
        "XGBoost": (XGBoostModel(), "xgboost.joblib"),
    }

    # Train and save each model
    for model_name, (model, filename) in models_config.items():
        print("\n" + "=" * 80)
        print(f"🚀 Training: {model_name}")
        print("=" * 80)

        try:
            # Train model
            print("⏳ Training in progress...")
            model.train(X_train, y_train)
            print("✅ Training completed!")

            # Evaluate on validation set
            print("📈 Evaluating on validation set...")
            metrics = model.evaluate(X_val, y_val)

            print("\n📊 Validation Metrics:")
            print(f" - Accuracy:  {metrics['accuracy']:.4f}")
            print(f" - Precision: {metrics['precision']:.4f}")
            print(f" - Recall:    {metrics['recall']:.4f}")
            print(f" - F1 Score:  {metrics['f1_score']:.4f}")
            if "auc" in metrics:
                print(f" - AUC:       {metrics['auc']:.4f}")

            # Save model to ./saved_models
            save_path = models_dir / filename
            model.save_model(save_path)
            print(f"\n💾 Model saved to: {save_path}")

        except Exception as e:
            print(f"❌ Error training {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("🎉 ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
    print(f"📁 Models saved in: {models_dir}")
    print("📌 Models can now be loaded from ./model/saved_models directory")
    print("=" * 80)


if __name__ == "__main__":
    train_and_save_all_models()
