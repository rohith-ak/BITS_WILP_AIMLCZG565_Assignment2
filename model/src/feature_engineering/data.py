import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .feature_engineering import feature_engineer


def load_dataset(file_path, use_feature_engineering=True):
    """
    Load and preprocess the Adult dataset.

    Args:
        file_path: Path to the CSV file
        use_feature_engineering: If True, apply advanced feature engineering.
                                 If False, use simple preprocessing (backward compatibility).

    Returns:
        X: Features DataFrame
        y: Target Series
    """
    if use_feature_engineering:
        # Use new feature engineering pipeline
        from .feature_engineering import load_and_engineer_dataset
        X, y, _ = load_and_engineer_dataset(file_path)
        return X, y
    else:
        # Original simple preprocessing (for backward compatibility)
        df = pd.read_csv(file_path)

        # Strip whitespace from column names
        df.columns = df.columns.str.strip()

        # Strip whitespace from all string columns
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip()

        # Handle missing values (represented as '?' in adult dataset)
        df = df.replace('?', pd.NA)
        df = df.dropna()

        # Encode categorical variables
        label_encoders = {}
        categorical_columns = df.select_dtypes(include=['object']).columns

        for col in categorical_columns:
            if col != 'income':  # Don't encode target yet
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                label_encoders[col] = le

        # Separate features and target variable
        X = df.drop(columns=['income'])

        # Encode target variable (income)
        le_target = LabelEncoder()
        y = le_target.fit_transform(df['income'])

        return X, y
