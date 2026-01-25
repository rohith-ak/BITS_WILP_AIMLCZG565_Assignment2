import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def feature_engineer(df):
    """
    Perform ENHANCED feature engineering on the Adult dataset.

    Args:
        df: Raw DataFrame loaded from CSV

    Returns:
        X: Feature-engineered DataFrame
        y: Target Series (binary: 0 for <=50K, 1 for >50K)
        sample_weight: Original fnlwgt values (optional)
    """

    # Create a copy to avoid modifying original
    df = df.copy()

    # Strip whitespace from all string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()

    # Replace '?' with NaN
    df.replace('?', np.nan, inplace=True)

    # Handle missing values - use mode for categorical
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # ==============================================================
    # 1. BASIC FEATURE ENGINEERING
    # ==============================================================

    # Target encoding (do this first)
    df['income_binary'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)

    # ==============================================================
    # 2. CAPITAL FEATURES - ENHANCED
    # ==============================================================

    df['has_capital_gain'] = (df['capital-gain'] > 0).astype(int)
    df['has_capital_loss'] = (df['capital-loss'] > 0).astype(int)
    df['has_any_capital'] = ((df['capital-gain'] > 0) | (df['capital-loss'] > 0)).astype(int)

    df['log_capital_gain'] = np.log1p(df['capital-gain'])
    df['log_capital_loss'] = np.log1p(df['capital-loss'])

    df['capital_net'] = df['capital-gain'] - df['capital-loss']
    df['log_capital_net'] = np.log1p(np.abs(df['capital_net']))

    df['capital_per_hour'] = df['capital_net'] / (df['hours-per-week'] + 1)

    # ==============================================================
    # 3. AGE FEATURES - ENHANCED
    # ==============================================================

    df['age_bucket'] = pd.cut(
        df['age'],
        bins=[0, 20, 30, 40, 50, 60, np.inf],
        labels=['17-20', '21-30', '31-40', '41-50', '51-60', '60+']
    )

    df['age_squared'] = df['age'] ** 2
    df['is_prime_age'] = ((df['age'] >= 25) & (df['age'] <= 55)).astype(int)
    df['is_young'] = (df['age'] < 30).astype(int)
    df['is_senior'] = (df['age'] >= 60).astype(int)

    # ==============================================================
    # 4. EDUCATION FEATURES - ENHANCED
    # ==============================================================

    def education_category(edu_num):
        if edu_num <= 8:
            return 'low'
        elif edu_num <= 12:
            return 'medium'
        elif edu_num <= 14:
            return 'high'
        else:
            return 'very_high'

    df['education_category'] = df['educational-num'].apply(education_category)
    df['has_advanced_degree'] = (df['educational-num'] >= 13).astype(int)
    df['has_college'] = (df['educational-num'] >= 13).astype(int)
    df['hs_or_less'] = (df['educational-num'] <= 12).astype(int)

    # ==============================================================
    # 5. WORK FEATURES - ENHANCED
    # ==============================================================

    df['hours_category'] = pd.cut(
        df['hours-per-week'],
        bins=[0, 30, 40, 50, 60, np.inf],
        labels=['part_time', 'normal', 'overtime', 'heavy_overtime', 'extreme']
    )

    df['works_overtime'] = (df['hours-per-week'] > 40).astype(int)
    df['works_extreme'] = (df['hours-per-week'] > 60).astype(int)
    df['is_part_time'] = (df['hours-per-week'] < 35).astype(int)
    df['work_intensity'] = (df['age'] * df['hours-per-week']) / 1000

    # ==============================================================
    # 6. INTERACTION FEATURES
    # ==============================================================

    df['age_education_interaction'] = df['age'] * df['educational-num']
    df['hours_education_interaction'] = df['hours-per-week'] * df['educational-num']
    df['age_hours_interaction'] = df['age'] * df['hours-per-week']
    df['education_capital_interaction'] = df['educational-num'] * df['log_capital_gain']

    # ==============================================================
    # 7. OCCUPATION & WORKCLASS FEATURES
    # ==============================================================

    professional_occupations = ['Exec-managerial', 'Prof-specialty', 'Tech-support']
    df['is_professional'] = df['occupation'].isin(professional_occupations).astype(int)
    df['is_private'] = (df['workclass'] == 'Private').astype(int)
    df['is_self_employed'] = df['workclass'].isin(['Self-emp-not-inc', 'Self-emp-inc']).astype(int)
    df['is_government'] = df['workclass'].str.contains('gov', case=False, na=False).astype(int)

    # ==============================================================
    # 8. MARITAL STATUS FEATURES
    # ==============================================================

    df['is_married'] = df['marital-status'].str.contains('Married', na=False).astype(int)
    df['never_married'] = (df['marital-status'] == 'Never-married').astype(int)

    # ==============================================================
    # 9. GENDER & RACE FEATURES
    # ==============================================================

    df['is_male'] = (df['gender'] == 'Male').astype(int)
    df['is_white'] = (df['race'] == 'White').astype(int)

    # ==============================================================
    # 10. COUNTRY FEATURES
    # ==============================================================

    df['is_usa'] = (df['native-country'] == 'United-States').astype(int)

    # ==============================================================
    # 11. RATIO & NORMALIZED FEATURES
    # ==============================================================

    df['education_age_ratio'] = df['educational-num'] / (df['age'] + 1)
    df['hours_age_ratio'] = df['hours-per-week'] / (df['age'] + 1)

    # ==============================================================
    # 12. LABEL ENCODING FOR CATEGORICAL VARIABLES
    # ==============================================================

    label_encoders = {}
    categorical_features = [
        'workclass', 'education', 'marital-status', 'occupation',
        'relationship', 'race', 'gender', 'native-country',
        'age_bucket', 'hours_category', 'education_category'
    ]

    for col in categorical_features:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    # ==============================================================
    # 13. SELECT FEATURES FOR MODEL
    # ==============================================================

    numerical_features = [
        'age', 'age_squared', 'fnlwgt', 'educational-num',
        'hours-per-week', 'capital-gain', 'capital-loss'
    ]

    log_features = [
        'log_capital_gain', 'log_capital_loss', 'log_capital_net'
    ]

    binary_features = [
        'has_capital_gain', 'has_capital_loss', 'has_any_capital',
        'is_prime_age', 'is_young', 'is_senior', 'has_advanced_degree',
        'has_college', 'hs_or_less', 'works_overtime', 'works_extreme',
        'is_part_time', 'is_professional', 'is_private', 'is_self_employed',
        'is_government', 'is_married', 'never_married', 'is_male',
        'is_white', 'is_usa'
    ]

    interaction_features = [
        'age_education_interaction', 'hours_education_interaction',
        'age_hours_interaction', 'education_capital_interaction',
        'work_intensity', 'capital_per_hour'
    ]

    ratio_features = [
        'education_age_ratio', 'hours_age_ratio', 'capital_net'
    ]

    encoded_features = [f'{col}_encoded' for col in categorical_features if f'{col}_encoded' in df.columns]

    all_features = (
        numerical_features +
        log_features +
        binary_features +
        interaction_features +
        ratio_features +
        encoded_features
    )

    all_features = [f for f in all_features if f in df.columns]

    X = df[all_features].copy()
    y = df['income_binary'].copy()
    sample_weight = df['fnlwgt'].copy()

    X.fillna(0, inplace=True)

    print("Feature engineering complete!")
    print(f"Total features: {X.shape[1]}")
    print("Feature categories:")
    print(f" - Numerical: {len(numerical_features)}")
    print(f" - Log-transformed: {len(log_features)}")
    print(f" - Binary flags: {len(binary_features)}")
    print(f" - Interactions: {len(interaction_features)}")
    print(f" - Ratios: {len(ratio_features)}")
    print(f" - Encoded categoricals: {len(encoded_features)}")

    return X, y, sample_weight


def load_and_engineer_dataset(file_path):
    """
    Load CSV and apply feature engineering.

    Args:
        file_path: Path to adult.csv

    Returns:
        X, y, sample_weight
    """
    df = pd.read_csv(file_path)
    return feature_engineer(df)
