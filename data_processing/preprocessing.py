import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, precision_score,
    recall_score, f1_score, confusion_matrix
)
# from xgboost import XGBClassifier


# Data Cleaning Function
def clean_data(df):
    # Step 1: Handle missing values
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric
        df[col].fillna(df[col].median(), inplace=True)  # Fill NaN with median

    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip()  # Strip whitespace
        df[col].fillna(df[col].mode()[0], inplace=True)  # Fill NaN with mode

    # Convert boolean columns to integers
    boolean_cols = ['senior_citizen', 'partner', 'dependents', 'phone_service', 'paperless_billing', 'churn']
    for col in boolean_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # One-hot encoding for categorical variables
    df = pd.get_dummies(df, drop_first=True)

    return df

# Data Splitting Function
def split_data(df):
    if 'churn' not in df.columns:
        raise KeyError("'churn' column not found in the dataset")
    
    X = df.drop('churn', axis=1)  # Features
    y = df['churn'].astype(int)  # Target (binary)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluation Function
def evaluate_model(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"Confusion Matrix: \n{cm}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(f"Precision: {precision_score(y_test, y_pred) * 100:.2f}%")
    print(f"Recall: {recall_score(y_test, y_pred) * 100:.2f}%")
    print(f"F1-Score: {f1_score(y_test, y_pred) * 100:.2f}%")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Random Forest Training Function
def train_random_forest(X_train, y_train, X_test, y_test):
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced',
        max_depth=10,
        min_samples_split=10
    )
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)

    print("Random Forest Evaluation:")
    evaluate_model(y_test, y_pred)
    return model

# XGBoost Training Function
def train_xgboost(X_train, y_train, X_test, y_test):
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    model = XGBClassifier(scale_pos_weight=20, random_state=42)
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)

    print("XGBoost Evaluation:")
    evaluate_model(y_test, y_pred)
    return model

# Adjust Threshold Function
def adjust_threshold(model, X_test, y_test, threshold=0.5):
    y_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities for "Yes"
    y_pred_adjusted = (y_prob >= threshold).astype(int)

    print(f"Evaluation with Threshold = {threshold}:")
    evaluate_model(y_test, y_pred_adjusted)

# Main Script
if __name__ == "__main__":
    # Example: Load your dataset
    # Replace with your actual dataset path
    data_path = "path_to_your_dataset.csv"
    df = pd.read_csv(data_path)
    
    # Clean data
    df_cleaned = clean_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df_cleaned)

    # Train Random Forest
    rf_model = train_random_forest(X_train, y_train, X_test, y_test)

    # Train XGBoost
    xgb_model = train_xgboost(X_train, y_train, X_test, y_test)

    # Adjust Threshold for Random Forest
    adjust_threshold(rf_model, X_test, y_test, threshold=0.7)
