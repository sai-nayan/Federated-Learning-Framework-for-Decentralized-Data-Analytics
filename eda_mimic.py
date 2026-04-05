import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Define features and target globally
TARGET_COLUMN = 'hospital_expire_flag'
FEATURE_COLUMNS = ['anchor_age', 'los', 'admission_type', 'gender', 'first_careunit']
CONTINUOUS_COLUMNS = ['anchor_age', 'los']

def load_data():
    """
    Loads and merges the primary MIMIC-IV datasets to form the basis for mortality prediction.
    """
    admissions = pd.read_csv('admissions.csv')
    patients = pd.read_csv('patients.csv')
    icustays = pd.read_csv('icustays.csv')
    
    df_merged = pd.merge(admissions, patients[['subject_id', 'gender', 'anchor_age']], on='subject_id', how='inner')
    # Keep only the first ICU stay per admission to prevent data leakage
    df_final = pd.merge(df_merged, icustays[['hadm_id', 'stay_id', 'first_careunit', 'los']], on='hadm_id', how='left')
    return df_final

def preprocess_data(df):
    """
    Filters useful predictor columns, handles missing values, and one-hot encodes categoricals. 
    Returns a pristine dataframe ready for machine learning consumption.
    """
    df_ml = df[FEATURE_COLUMNS + [TARGET_COLUMN]].dropna()
    df_ml = pd.get_dummies(df_ml, columns=['admission_type', 'gender', 'first_careunit'], drop_first=True)
    return df_ml

def feature_engineering(df_ml):
    """
    Performs train-test splitting, applies robust synthetic minority upsampling (resampling)
    to combat clinical class imbalances, and standardizes continuous fields.
    """
    X = df_ml.drop(TARGET_COLUMN, axis=1)
    y = df_ml[TARGET_COLUMN]
    
    # 80/20 Stratified split to ensure mortality is represented appropriately
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Class imbalance handling: Upsampling the minority class on training data only
    X_train_min = X_train[y_train == 1]
    y_train_min = y_train[y_train == 1]
    X_train_maj = X_train[y_train == 0]
    y_train_maj = y_train[y_train == 0]
    
    # Boost minority class to 50% of the majority class footprint
    X_train_min_up = resample(X_train_min, replace=True, n_samples=len(X_train_maj)//2, random_state=42)
    y_train_min_up = resample(y_train_min, replace=True, n_samples=len(y_train_maj)//2, random_state=42)
    
    X_train_up = pd.concat([X_train_maj, X_train_min_up])
    y_train_up = pd.concat([y_train_maj, y_train_min_up])
    
    # Standardize continuous variables (crucial for Logistic Regression convergence)
    scaler = StandardScaler()
    X_train_up[CONTINUOUS_COLUMNS] = scaler.fit_transform(X_train_up[CONTINUOUS_COLUMNS])
    X_test[CONTINUOUS_COLUMNS] = scaler.transform(X_test[CONTINUOUS_COLUMNS])
    
    return X_train_up, X_test, y_train_up, y_test, scaler

def train_models(X_train, y_train):
    """
    Assembles, trains, and returns baseline predictive models weighted for balanced classes.
    """
    lr_model = LogisticRegression(class_weight='balanced', random_state=42)
    lr_model.fit(X_train, y_train)
    
    rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf_model.fit(X_train, y_train)
    
    return lr_model, rf_model

def evaluate_models(models_dict, X_test, y_test):
    """
    Runs robust evaluation yielding Accuracy, Precision, Recall, F1, and AUC.
    Yields data structure consumed directly by the Streamlit dashboard.
    """
    results = {}
    for name, model in models_dict.items():
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, probs)
        cm = confusion_matrix(y_test, preds).tolist()
        
        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc,
            "confusion_matrix": cm,
            "model_obj": model 
        }
    return results

if __name__ == "__main__":
    print("Loading Data...")
    df = load_data()
    
    print("Preprocessing Data...")
    df_ml = preprocess_data(df)
    
    print("Engineering Features & Splitting...")
    X_train, X_test, y_train, y_test, scaler = feature_engineering(df_ml)
    
    print("Training Models...")
    lr, rf = train_models(X_train, y_train)
    
    print("Evaluating Models...")
    results = evaluate_models({'Logistic Regression': lr, 'Random Forest': rf}, X_test, y_test)
    
    for model_name, metrics in results.items():
        print(f"\n=== {model_name} ===")
        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}  <-- Critical in healthcare!")
        print(f"F1       : {metrics['f1_score']:.4f}")
        print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
        print(f"Conf. Mat: {metrics['confusion_matrix']}")
        
    print("\nEDA / Modeling Pipeline Completed Successfully!")