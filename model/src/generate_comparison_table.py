"""
Script to generate comparison table for all 6 ML models.
This script trains all models and generates a comprehensive comparison table.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys
from tabulate import tabulate

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


def generate_comparison_table(data_path, use_feature_engineering=True, test_size=0.2, random_state=42):
    """
    Generate comparison table for all models.

    Args:
        data_path: Path to the dataset csv file
        use_feature_engineering: Whether to use advanced feature engineering
        test_size: Proportion of test set
        random_state: Random seed for reproducibility

    Returns:
        comparison_df: DataFrame with comparison metrics
    """
    print("=" * 80)
    print("ADULT CENSUS INCOME CLASSIFICATION - MODEL COMPARISON")
    print("=" * 80)
    print(f"Dataset: {data_path}")
    print(f"Feature Engineering: {'Enabled' if use_feature_engineering else 'Disabled'}")
    print(f"Test size: {test_size * 100}%")
    print(f"Random State: {random_state}")
    print("\n" + "=" * 80)

    # Load and preprocess data
    print("\n📊 Loading and preprocessing data...")
    X, y = load_dataset(data_path, use_feature_engineering=use_feature_engineering)

    print(f"✅ Data loaded successfully")
    print(f" - Features shape: {X.shape}")
    print(f" - Target shape: {y.shape}")
    print(f" - Class distribution: {pd.Series(y).value_counts().to_dict()}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"\n📊 Data split:")
    print(f" - Training samples: {X_train.shape[0]}")
    print(f" - Testing samples: {X_test.shape[0]}")

    # Define models
    models = {
        "Logistic Regression": LogisticRegressionModel(),
        "Decision Tree": DecisionTreeModel(),
        "kNN": KNNModel(),
        "Naive Bayes": NaiveBayesModel(),
        "Random Forest (Ensemble)": RandomForestModel(),
        "XGBoost (Ensemble)": XGBoostModel(),
    }

    # Store results
    results = []

    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"\n{'=' * 80}")
        print(f"🚀 Training: {model_name}")
        print(f"{'=' * 80}")

        try:
            # Train model
            print("⏳ Training in progress...")
            model.train(X_train, y_train)
            print("✅ Training completed!")

            # Evaluate model
            print("📈 Evaluating model...")
            metrics = model.evaluate(X_test, y_test)
            print("✅ Evaluation completed!")

            # Store results
            result = {
                "ML Model Name": model_name,
                "Accuracy": metrics["accuracy"],
                "AUC": metrics.get("auc", "N/A"),
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1_score"],
                "MCC": metrics["mcc"]
            }
            results.append(result)

            # Print metrics
            print(f"\n📊 Metrics for {model_name}:")
            print(f" - Accuracy: {metrics['accuracy']:.4f}")
            
            auc_value = metrics.get("auc", "N/A")
            if isinstance(auc_value, str):
                print(f" - AUC: {auc_value}")
            else:
                print(f" - AUC: {auc_value:.4f}")


            print(f" - Precision: {metrics['precision']:.4f}")
            print(f" - Recall: {metrics['recall']:.4f}")
            print(f" - F1 Score: {metrics['f1_score']:.4f}")
            print(f" - MCC: {metrics['mcc']:.4f}")

        except Exception as e:
            print(f"❌ Error training {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()

            result = {
                "ML Model Name": model_name,
                "Accuracy": "Error",
                "AUC": "Error",
                "Precision": "Error",
                "Recall": "Error",
                "F1": "Error",
                "MCC": "Error"
            }
            results.append(result)

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)

    # Round numeric columns
    numeric_columns = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    for col in numeric_columns:
        if col in comparison_df.columns:
            comparison_df[col] = comparison_df[col].apply(
                lambda x: round(x, 4) if isinstance(x, (int, float)) else x
            )

    return comparison_df


def save_comparison_table(comparison_df, output_dir=None):
    """
    Save comparison table to CSV and display as formatted table.

    Args:
        comparison_df: DataFrame with comparison metrics
        output_dir: Directory to save results (defaults to model/results)
    """
    if output_dir is None:
        output_dir = project_root / "model" / "results"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True)

    # Save to CSV
    csv_path = output_dir / "model_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"\n📁 Comparison table saved to: {csv_path}")

    # Save to formatted text file
    txt_path = output_dir / "model_comparison.txt"
    with open(txt_path, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("ADULT CENSUS INCOME CLASSIFICATION - MODEL COMPARISON TABLE\n")
        f.write("=" * 100 + "\n\n")
        f.write(tabulate(comparison_df, headers="keys", tablefmt="grid", showindex=False))
        f.write("\n\n" + "=" * 100 + "\n")

    print(f"📁 Formatted table saved to: {txt_path}")

    # Display formatted table in console
    print("\n" + "=" * 100)
    print("COMPARISON TABLE")
    print("=" * 100)
    print(tabulate(comparison_df, headers="keys", tablefmt="grid", showindex=False))
    print("\n" + "=" * 100)

    # Best models by metric
    print("\n🏆 BEST MODELS BY METRIC:")
    print("=" * 100)

    numeric_metrics = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    for metric in numeric_metrics:
        if metric in comparison_df.columns:
            numeric_rows = comparison_df[comparison_df[metric].apply(lambda x: isinstance(x, (int, float)))]
            if not numeric_rows.empty:
                best_idx = numeric_rows[metric].idxmax()
                best_model = comparison_df.loc[best_idx, "ML Model Name"]
                best_value = comparison_df.loc[best_idx, metric]
                print(f"{metric:12}: {best_model:30} ({best_value:.4f})")

    print("=" * 100)

    # LaTeX output
    latex_path = output_dir / "model_comparison_latex.txt"
    with open(latex_path, "w") as f:
        f.write(tabulate(comparison_df, headers="keys", tablefmt="latex", showindex=False))
    print(f"\n📄 LaTeX table saved to: {latex_path}")

    # HTML output
    html_path = output_dir / "model_comparison.html"
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
<title>Model Comparison - Adult Census Income Classification</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 40px;
    background-color: #f5f5f5;
}}
h1 {{
    color: #333;
    text-align: center;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    background-color: white;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}}
th {{
    background-color: #4CAF50;
    color: white;
    padding: 12px;
    text-align: left;
}}
td {{
    padding: 10px;
    border-bottom: 1px solid #ddd;
}}
tr:hover {{
    background-color: #f5f5f5;
}}
.best {{
    background-color: #d4edda;
    font-weight: bold;
}}
</style>
</head>
<body>
<h1>Adult Census Income Classification - Model Comparison</h1>
{comparison_df.to_html(index=False, classes="table", border=0)}
</body>
</html>
"""
    with open(html_path, "w") as f:
        f.write(html_content)

    print(f"🌐 HTML table saved to: {html_path}")


def main():
    """Main function to run the comparison."""
    data_path = project_root / "model" / "data" / "adult.csv"

    if not data_path.exists():
        print(f"❌ Error: Data file not found at {data_path}")
        print(f"   Looking at: {data_path.absolute()}")
        print("\nExpected directory structure:")
        print(" project-folder/")
        print(" ├── model/")
        print(" │   ├── data/")
        print(" │   │   └── adult.csv  <- Should be here")
        print(" │   └── src/")
        print(" │       └── generate_comparison_table.py  <- You are here")
        return

    comparison_df = generate_comparison_table(
        data_path=data_path,
        use_feature_engineering=True,
        test_size=0.2,
        random_state=42
    )

    save_comparison_table(comparison_df)

    print("\n✅ Comparison table generation completed successfully!")
    print(f"📁 All results saved to: {project_root / 'model' / 'results'}")


if __name__ == "__main__":
    main()
