import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import time
from sklearn.metrics import roc_curve

from eda_mimic import load_data, preprocess_data, feature_engineering, train_models, evaluate_models, FEATURE_COLUMNS, CONTINUOUS_COLUMNS
from federated_nn import run_simulation as run_nn_simulation

st.set_page_config(page_title="Healthcare AI Hub", page_icon="🏥", layout="wide")

st.markdown("""
<style>
    .metric-card {
        background-color: var(--background-color);
        border: 1px solid var(--secondary-text-color);
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-label {
        font-size: 1.1rem;
        color: #888;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
    }
    .pulse-node {
        width: 25px;
        height: 25px;
        border-radius: 50%;
        display: inline-block;
        background-color: #10b981;
        animation: pulse 2s infinite;
        margin: 10px;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_prep_all():
    df = load_data()
    df_ml = preprocess_data(df)
    X_train, X_test, y_train, y_test, scaler = feature_engineering(df_ml)
    lr, rf = train_models(X_train, y_train)
    results = evaluate_models({'Logistic Regression': lr, 'Random Forest': rf}, X_test, y_test)
    return df, df_ml, X_train, X_test, y_train, y_test, scaler, lr, rf, results

with st.spinner("Initializing Clinical ML Pipeline (Loading & Processing eda_mimic.py)..."):
    try:
        df, df_ml, X_train, X_test, y_train, y_test, scaler, lr, rf, results = load_and_prep_all()
    except Exception as e:
        st.error(f"Error loading system: {e}. Please ensure admissions.csv, patients.csv, and icustays.csv exist.")
        st.stop()


st.sidebar.title("🏥 Navigation")
nav = st.sidebar.radio("Go to:", [
    "📊 Project Overview",
    "📂 Data Summary",
    "⚙️ Preprocessing Pipeline",
    "📈 Model Performance",
    "📉 Confusion Matrix",
    "📊 ROC Curve",
    "🌲 Feature Importance",
    "🧠 Federated Learning",
    "🧪 Live Prediction",
    "📌 Final Insights"
])

if nav == "📊 Project Overview":
    st.title("📊 Project Overview: ICU Mortality Prediction")
    st.write("---")
    st.markdown("""
    ### Problem Statement
    In clinical settings, identifying high-risk patients early is crucial for resource allocation and preemptive care. 
    This system predicts **hospital mortality** for patients admitted to the ICU.
    
    ### The Dataset
    We utilize a subset of the prestigious **MIMIC-IV** dataset, integrating patient demographics, admission information, and ICU stay characteristics.
    
    ### Why Recall Matters Most
    > **⚠️ Clinical Alert**  
    > In healthcare, **Recall (Sensitivity)** is prioritized over raw Accuracy. A false negative (failing to identify a patient at risk of dying) carries devastating consequences compared to a false positive (which may simply result in extra monitoring).
    """)
    
elif nav == "📂 Data Summary":
    st.title("📂 Data Summary")
    st.write("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Raw Patient Records", df.shape[0])
    col2.metric("Extracted Features", len(FEATURE_COLUMNS))
    col3.metric("Mortality Rate", f"{(df['hospital_expire_flag'].mean()*100):.2f}%")
    
    st.subheader("Raw Data Sample")
    st.dataframe(df.head(15))
    
    st.subheader("Core Features Used")
    st.write([col for col in FEATURE_COLUMNS])

elif nav == "⚙️ Preprocessing Pipeline":
    st.title("⚙️ Preprocessing Pipeline")
    st.write("---")
    
    st.markdown("""
    To train robust models, we execute a rigorous preprocessing pipeline rooted in `eda_mimic.py`:
    1. **Data Integration**: Securely joining Admissions, Patients, and ICU stays.
    2. **Data Cleaning**: Rows with missing core features are dropped.
    3. **Encoding**: Categorical variables (Gender, Admission Type, ICU Unit) are converted to multiple numeric columns using One-Hot Encoding.
    4. **Standardization**: Continuous variables (`anchor_age`, `los`) are scaled to center bounds relying on standard deviations.
    5. **Imbalance Handling**: The minority class (Mortality) is synthetically upsampled during training to prevent the model from blindly predicting survival.
    """)
    st.subheader("Preprocessed Feature Space (Ready for ML)")
    st.dataframe(df_ml.head(10))

elif nav == "📈 Model Performance":
    st.title("📈 Model Performance (Centralized)")
    st.write("---")
    
    for model_name, metrics in results.items():
        st.subheader(f"Model: {model_name}")
        cols = st.columns(5)
        cols[0].metric("Accuracy", f"{metrics['accuracy']:.3f}")
        cols[1].metric("Precision", f"{metrics['precision']:.3f}")
        cols[2].metric("Recall", f"{metrics['recall']:.3f}")
        cols[3].metric("F1-Score", f"{metrics['f1_score']:.3f}")
        cols[4].metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
        st.write("")

elif nav == "📉 Confusion Matrix":
    st.title("📉 Confusion Matrix Analysis")
    st.write("---")
    st.markdown("Visualizing True Positives vs False Negatives is critical for clinical adoption. Notice how our balanced models minimize False Negatives.")
    
    col1, col2 = st.columns(2)
    def plot_cm(cm, title):
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual (0=Survive, 1=Die)')
        return fig
        
    with col1:
        st.pyplot(plot_cm(results['Logistic Regression']['confusion_matrix'], "Logistic Regression"))
    with col2:
        st.pyplot(plot_cm(results['Random Forest']['confusion_matrix'], "Random Forest"))

elif nav == "📊 ROC Curve":
    st.title("📊 Receiver Operating Characteristic (ROC)")
    st.write("---")
    st.markdown("""
    **Why ROC-AUC?** Accuracy is misleading when approx 90% of patients survive. ROC-AUC evaluates the model's true capability to distinguish classes cleanly across *all* probability thresholds.
    """)
    
    fig, ax = plt.subplots(figsize=(8,6))
    for model_name, metrics in results.items():
        probs = metrics['model_obj'].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax.plot(fpr, tpr, label=f"{model_name} (AUC = {metrics['roc_auc']:.3f})", lw=2)
        
    ax.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve Comparison')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        st.pyplot(fig)

elif nav == "🌲 Feature Importance":
    st.title("🌲 Model Interpretability: Feature Importance")
    st.write("---")
    st.markdown("Doctors need to know *why* the model flags a patient. Here is the feature influence learned natively by the Random Forest.")
    
    rf_model = results['Random Forest']['model_obj']
    importances = rf_model.feature_importances_
    features = X_train.columns
    
    fi_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    fi_df = fi_df.sort_values(by='Importance', ascending=False).head(15)
    
    chart = alt.Chart(fi_df).mark_bar(color='#10b981').encode(
        x=alt.X('Importance:Q', title='Relative Importance (%)'),
        y=alt.Y('Feature:N', sort='-x', title='Predictor (Included Extracted Dummies)'),
        tooltip=['Feature', 'Importance']
    ).properties(height=500).interactive()
    
    st.altair_chart(chart, use_container_width=True)

elif nav == "🧠 Federated Learning":
    st.title("🧠 Privacy-Preserving Federated Learning")
    st.write("---")
    st.markdown("""
    ### Why Federated Learning in Healthcare?
    Patient data is sensitive and legally protected under regulations like HIPAA and GDPR. Federated Learning allows multiple hospital datacenters to train a global Deep ResNet mortality predictor **collaboratively without ever transferring raw patient data**. 
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("FL Configuration")
    num_rounds = st.sidebar.slider("Communication Rounds", 1, 10, 3)
    num_clients = st.sidebar.slider("Number of Hospitals", 2, 5, 3)
    
    st.markdown(f"#### Active Federated Hospital Nodes: {num_clients}")
    nodes_html = "".join([f"<div class='pulse-node' style='animation-delay: {i*0.2}s' title='Hospital {i+1}'></div>" for i in range(num_clients)])
    st.markdown(nodes_html, unsafe_allow_html=True)
    
    if st.button("Start Federated Deep ResNet Training"):
        with st.spinner("Connecting Distributed Clients and Initializing FL Rounds..."):
            fl_results = run_nn_simulation(num_rounds, num_clients)
            
        st.success("Federated Training Execution Complete!")
        
        metrics = fl_results['metrics']
        cols = st.columns(4)
        cols[0].metric("Global Accuracy", f"{metrics['Accuracy']:.3f}")
        cols[1].metric("Global Precision", f"{metrics['Precision']:.3f}")
        cols[2].metric("Global Recall", f"{metrics['Recall']:.3f}")
        cols[3].metric("Global ROC-AUC", f"{metrics['ROC-AUC']:.3f}")
        
elif nav == "🧪 Live Prediction":
    st.title("🧪 Live Clinical Prediction Interface")
    st.write("---")
    st.markdown("Enter patient admission details here and run a live mortality probabilistic inference. This backend runs entirely on our strictly synchronized `eda_mimic.py` pipeline.")
    
    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Patient Age (anchor_age)", min_value=18, max_value=120, value=65)
            los = st.number_input("Prior ICU Length of Stay (days)", min_value=0.0, max_value=200.0, value=3.5)
            gender = st.selectbox("Gender", ["M", "F"])
        with col2:
            admission_type = st.selectbox("Admission Type", ["URGENT", "EW EMER.", "ELECTIVE", "DIRECT OBSERVATION", "EU OBSERVATION"])
            careunit = st.selectbox("First Care Unit", ["Medical Intensive Care Unit (MICU)", "Cardiac Vascular Intensive Care Unit (CVICU)", "Surgical Intensive Care Unit (SICU)", "Trauma SICU (TSICU)", "Coronary Care Unit (CCU)"])
        
        submit = st.form_submit_button("Assess Mortality Risk")
        
    if submit:
        sample_df = pd.DataFrame(0, index=[0], columns=X_train.columns)
        
        cont_vals = pd.DataFrame([[age, los]], columns=CONTINUOUS_COLUMNS)
        scaled_vals = scaler.transform(cont_vals)
        sample_df['anchor_age'] = scaled_vals[0, 0]
        sample_df['los'] = scaled_vals[0, 1]
        
        if f"gender_{gender}" in sample_df.columns:
            sample_df[f"gender_{gender}"] = 1
        if f"admission_type_{admission_type}" in sample_df.columns:
            sample_df[f"admission_type_{admission_type}"] = 1
        if f"first_careunit_{careunit}" in sample_df.columns:
            sample_df[f"first_careunit_{careunit}"] = 1
            
        prob = results['Random Forest']['model_obj'].predict_proba(sample_df)[0][1]
        
        st.markdown("---")
        if prob >= 0.5:
            st.error(f"🚨 **HIGH RISK**: This patient has a **{prob*100:.1f}%** localized probability of non-survival.")
            st.progress(prob)
        else:
            st.success(f"✅ **LOWER RISK**: This patient has a **{prob*100:.1f}%** localized probability of non-survival.")
            st.progress(prob)
            
elif nav == "📌 Final Insights":
    st.title("📌 Clinical System Insights")
    st.write("---")
    
    st.markdown("""
    ### Why Random Forest Outperforms Logistic Regression
    The healthcare data environment is highly non-linear. Interactions between patient variables (e.g., an elderly patient in the CVICU vs a young patient in the MICU) are dynamically captured by Random Forest tree depth branching splits. Conversely, Logistic Regression erroneously assumes independence across these features.
    
    ### Diagnostic Efficacy
    - Our **Recall-centric optimization framework** mitigates deadly False Negatives without destroying True Positive identification protocols.
    - An elevated **ROC-AUC** across centralized models demonstrates robust predictive threshold signaling despite extreme survival-class skew.
    - Seamlessly porting the exact same rigorous standard scaler & identical feature dimensions to an encoded **Federated Learning ResNet** guarantees scaling capabilities for massive health consortia.
    
    > *End of Output Summary Architecture.*
    """)
