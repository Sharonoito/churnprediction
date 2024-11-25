import os
from django.conf import settings
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
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
        # df[col] = df[col].astype(str).str.strip()  # Strip whitespace
        df[col] = df[col].astype('category')
        df[col].fillna(df[col].mode()[0], inplace=True)  # Fill NaN with mode

        # Limit the number of categories by grouping rare values into 'Other'
        df = reduce_categories(df, col)

    # Convert boolean columns to integers
    boolean_cols = ['senior_citizen', 'partner', 'dependents', 'phone_service', 'paperless_billing', 'churn']
    for col in boolean_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # One-hot encoding for categorical variables

    # df = pd.get_dummies(df, drop_first=True)
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, sparse=True)


    return df

# Function to reduce the number of categories in categorical columns
def reduce_categories(df, col, threshold=100):
    """Group infrequent categories into 'Other'."""
    value_counts = df[col].value_counts()
    to_replace = value_counts[value_counts < threshold].index
    df[col] = df[col].replace(to_replace, 'Other')
    return df


# Data Splitting Function
def split_data(df):
    if 'churn' not in df.columns:
        raise KeyError("'churn' column not found in the dataset")
    
    X = df.drop('churn', axis=1)  # Features
    y = df['churn'].astype(int)  # Target (binary)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluation Function
def evaluate_model(X_train, y_train, X_test, y_test, model_type='RandomForest'):
    if model_type == 'RandomForest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'LogisticRegression':
        model = LogisticRegression(random_state=42)
    elif model_type == 'XGBoost':
        model = XGBClassifier(random_state=42)
    else:
        raise ValueError("Unsupported model type")

    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_type} Model Accuracy: {accuracy * 100:.2f}%")

    return model

# Save the trained model
def save_model(model, model_name='churn_model'):   

    model_path = os.path.join(settings.BASE_DIR, 'models', f'{model_name}.pkl')

    # Save the model to a file
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


def load_model(model_name='churn_model'):
    model_path = os.path.join(settings.BASE_DIR, 'models', f'{model_name}.pkl')

    # Check if the model exists
    if os.path.exists(model_path):
        try:
            # Try to load the model
            model = joblib.load(model_path)
            return model
        except EOFError:
            # If EOFError occurs (corrupted model file), raise an error
            raise EOFError("Model file appears to be corrupted. Please re-train the model.")
    else:
        raise FileNotFoundError("Model not found. Please ensure the model is saved and available.")


def align_columns(df, model):
    # Get the features that the model was trained on (stored in the model)
    model_features = model.feature_names_in_  # This retrieves the original feature names from the trained model

    # Add missing columns with default values (e.g., 0)
    for col in model_features:
        if col not in df.columns:
            df[col] = 0  # Default value for missing columns
    
    df = df[model_features]  # Reorder columns to match the model's expected input order

    return df


def preprocess_for_prediction(data):
    # Assuming `data` is a dictionary with relevant features
    df = pd.DataFrame([data])

    # Handle categorical columns (same as cleaning)
    categorical_cols = ['monthly_charges', 'PaymentMethod', 'PaperlessBilling']  # Example, replace with your actual categorical columns
    for col in categorical_cols:
        df[col] = df[col].astype('category')
        df[col].fillna(df[col].mode()[0], inplace=True)
        df = reduce_categories(df, col)

    # Convert categorical columns to dummy variables (sparse)
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, sparse=True)

    return df


# Prediction function
def predict(gender, SeniorCitizen, Partner, Dependents, PaperlessBilling,PaymentMethod,tenure, monthly_charges,  total_charges, Churn,PhoneService, MultipleLines, InternetService, OnlineSecurity,OnlineBackup, DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract, model_type='RandomForest'):
    # Create a dictionary of the transaction data
    transaction_data = {
        'gender': gender,
        'senior_citizen': SeniorCitizen,
        'partner': Partner,
        'dependents': Dependents,
        'tenure': tenure,
        'phone_service': PhoneService,
        'multiple_lines':MultipleLines,
        'internet_service':InternetService,
        'online_security':OnlineSecurity,
        'online_backup': OnlineBackup,
        'device_protection': DeviceProtection,
        'tech_support':TechSupport,
        'streaming_tv': StreamingTV,
        'streaming_movies':StreamingMovies,
        'contract': Contract,
        'paperless_billing': PaperlessBilling,
        'payment_method': PaymentMethod,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'churn': Churn,

    }
    

    # Load the model
    model = load_model(model_name=model_type)

    # Preprocess the data (no scaler needed)
    processed_data = preprocess_for_prediction(transaction_data)

    # Make the prediction
    prediction = model.predict(processed_data)

    # Return the prediction result (churn or not)
    return "Churn" if prediction[0] == 1 else "Non-Churn"

# def adjust_threshold(model, X_test, y_test, threshold=0.5):
#     y_prob = model.predict_proba(X_test)[:, 1]  
#     y_pred_adjusted = (y_prob >= threshold).astype(int)

#     print(f"Evaluation with Threshold = {threshold}:")
#     evaluate_model(y_test, y_pred_adjusted)


