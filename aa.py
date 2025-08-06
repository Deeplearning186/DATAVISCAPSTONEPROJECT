# app.py
# Standalone Dash application for Telco Customer Churn Analytics

import os
import socket
import pandas as pd
import numpy as np
import warnings

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

warnings.filterwarnings('ignore')

# ============================================
# DATA LOADING AND PREPARATION
# ============================================

def load_data(path="TelcoCustomerChurn.csv"):
    """Load and clean Telco Customer Churn dataset, or generate sample data if not found."""
    try:
        df = pd.read_csv(path)
        print(f"✅ Data loaded successfully: {len(df)} records")
    except FileNotFoundError:
        print("⚠️ TelcoCustomerChurn.csv not found. Creating sample data...")
        np.random.seed(42)
        n_samples = 1000
        df = pd.DataFrame({
            'customerID': [f'CUST{i:04d}' for i in range(n_samples)],
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'Partner': np.random.choice(['Yes', 'No'], n_samples),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples),
            'tenure': np.random.randint(0, 73, n_samples),
            'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.3, 0.2]),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
            'PaymentMethod': np.random.choice([
                'Electronic check', 'Mailed check',
                'Bank transfer (automatic)', 'Credit card (automatic)'
            ], n_samples),
            'MonthlyCharges': np.random.uniform(20, 120, n_samples),
            'TotalCharges': np.random.uniform(20, 8000, n_samples),
            'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.27, 0.73])
        })

    # Clean TotalCharges
    if df['TotalCharges'].dtype == object:
        df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['MonthlyCharges'], inplace=True)

    # Feature engineering
    df['AvgChargesPerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72],
                                labels=['0-1 Year', '1-2 Years', '2-4 Years', '4+ Years'])
    df['MonthlyChargesGroup'] = pd.cut(df['MonthlyCharges'], bins=[0, 35, 65, 95, 120],
                                       labels=['Low', 'Medium', 'High', 'Very High'])
    return df

# ============================================
# MACHINE LEARNING PREPARATION
# ============================================

def prepare_ml_data(df):
    """Encode and scale features for ML models."""
    ml_df = df.drop(['customerID', 'TenureGroup', 'MonthlyChargesGroup'], axis=1, errors='ignore').copy()
    label_encoders = {}
    for col in ml_df.select_dtypes(include=['object']).columns:
        if col != 'Churn':
            le = LabelEncoder()
            ml_df[col] = le.fit_transform(ml_df[col])
            label_encoders[col] = le

    X = ml_df.drop('Churn', axis=1)
    y = (ml_df['Churn'] == 'Yes').astype(int)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, X.columns

# Load data and train models

df = load_data()
X, y, feature_names = prepare_ml_data(df)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}
model_results = {}
print("🤖 Training machine learning models...")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    model_results[name] = {
        'model': model,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'y_pred': y_pred,
        'y_pred_proba': y_proba
    }
    print(f"✅ {name} trained - Accuracy: {model_results[name]['accuracy']:.3f}")

# ============================================
# INITIALIZE DASH APP
# ============================================

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Custom styles
custom_css = {
    'container': {'padding': '20px', 'backgroundColor': '#f8f9fa'},
    'header': {'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'},
    'card': {'backgroundColor': 'white', 'borderRadius': '10px', 'padding': '20px',
             'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)', 'marginBottom': '20px'}
}

# Layout and callbacks omitted for brevity; insert layout and all callback definitions here
# ...

# ============================================
# RUN SERVER
# ============================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    try:
        app.run_server(debug=True, port=port)
    except OSError:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            free_port = s.getsockname()[1]
        print(f"Port {port} in use, switching to {free_port}")
        app.run_server(debug=True, port=free_port)
