import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import time
from sklearn.metrics import roc_curve

st.set_page_config(page_title="Federated AI Hub", page_icon="🌐", layout="wide")

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
    .metric-value { font-size: 2.2rem; font-weight: bold; }
    .pulse-node {
        width: 25px; height: 25px; border-radius: 50%; display: inline-block;
        background-color: #10b981; animation: pulse 2s infinite; margin: 10px;
    }
    .pulse-node-active { background-color: #f59e0b; }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    .log-box { font-family: monospace; font-size: 0.9em; background-color: #1e1e1e; color: #a3e635; padding: 10px; border-radius: 5px; height: 250px; overflow-y: auto; }
</style>
""", unsafe_allow_html=True)

# MULTI-DOMAIN LOADER
@st.cache_resource
def load_domain_assets(domain_name):
    if "Healthcare" in domain_name:
        import eda_mimic
        df = eda_mimic.load_data()
        df_ml = eda_mimic.preprocess_data(df)
        X_tr, X_te, y_tr, y_te, scl = eda_mimic.feature_engineering(df_ml)
        lr, rf = eda_mimic.train_models(X_tr, y_tr)
        res = eda_mimic.evaluate_models({'Logistic Regression': lr, 'Random Forest': rf}, X_te, y_te)
        from federated_nn import run_simulation
        return df, X_tr, X_te, y_tr, y_te, scl, res, run_simulation, "Healthcare", True
    elif "Finance" in domain_name:
        import federated_finance
        df, X_tr, X_te, y_tr, y_te, scl, res = federated_finance.get_centralized_metrics()
        from federated_finance import run_simulation
        return df, X_tr, X_te, y_tr, y_te, scl, res, run_simulation, "Finance", False
    else:
        import federated_student
        df, X_tr, X_te, y_tr, y_te, scl, res = federated_student.get_centralized_metrics()
        from federated_student import run_simulation
        return df, X_tr, X_te, y_tr, y_te, scl, res, run_simulation, "Student", False

# TOP DROPDOWN: Domain Select
c1, c2 = st.columns([8, 2])
with c1: st.title("🌐 Universal Federated Learning Dashboard")
with c2: domain_selection = st.selectbox("Select Domain", [
    "Healthcare (MIMIC-IV)", 
    "Finance (German Credit)", 
    "Student Performance"
])

with st.spinner(f"Loading {domain_selection} Models & Data..."):
    df, X_train, X_test, y_train, y_test, scaler, results, sim_func, d_type, show_roc = load_domain_assets(domain_selection)

if results is None:
    st.error("Error formatting datasets for this domain perfectly. Fallback required.")
    st.stop()

# SIDEBAR NAVIGATION
st.sidebar.title("Navigation")
nav = st.sidebar.radio("Go to:", [
    "🧠 Overview",
    "📊 Centralized Models",
    "🌐 Federated Learning",
    "🧪 Live Prediction"
])

st.write("---")

if nav == "🧠 Overview":
    st.header(f"🧠 {d_type} Analysis Overview")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        **Domain Objective:**
        - {"Predicting ICU Mortality to prioritize clinical resources." if d_type=="Healthcare" else "Predicting Credit Risk / Default probability." if d_type=="Finance" else "Predicting Academic Pass status for early intervention."}
        - **Data Shape**: `{df.shape[0]} records`, `{df.shape[1]} raw columns`
        - {"**Critical Note**: We optimize heavily for **Recall** in healthcare to minimize false negatives (missed deaths)." if d_type=="Healthcare" else "Standard balanced accuracy is maintained."}
        """)
    with c2:
        if d_type == "Healthcare":
            st.metric("Mortality Rate (Imbalance)", f"{(df['hospital_expire_flag'].mean()*100):.1f}%")
        else:
            st.metric("Dataset Head", f"{df.shape[0]} entities")
    st.dataframe(df.head(10), use_container_width=True)

elif nav == "📊 Centralized Models":
    st.header(f"📊 Centralized Models ({d_type})")
    
    for model_name, metrics in results.items():
        st.subheader(f"⚡ {model_name}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        c2.metric("Precision", f"{metrics['precision']:.3f}")
        c3.metric("Recall", f"{metrics['recall']:.3f}")
        c4.metric("F1-Score", f"{metrics['f1_score']:.3f}")
        
        c_fig1, c_fig2 = st.columns(2)
        with c_fig1:
            fig, ax = plt.subplots(figsize=(4,3))
            sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)
            
        with c_fig2:
            if show_roc and "model_obj" in metrics and hasattr(metrics['model_obj'], "predict_proba"):
                probs = metrics['model_obj'].predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, probs)
                fig_roc, ax_roc = plt.subplots(figsize=(4,3))
                ax_roc.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.2f}", color='#8b5cf6')
                ax_roc.plot([0, 1], [0, 1], 'k--')
                ax_roc.set_title("ROC Curve")
                ax_roc.legend()
                st.pyplot(fig_roc)
            else:
                st.info("ROC-AUC tracking disabled for native ensemble models in this domain.")

elif nav == "🌐 Federated Learning":
    st.header(f"🌐 Real-Time Federated Learning ({d_type})")
    st.markdown("Federated edge nodes stream gradients back to the central orchestrator asynchronously without exposing raw PII data.")
    
    # Sim Config
    c1, c2, c3 = st.columns(3)
    num_rounds = c1.slider("Comm Rounds", 1, 15, 5)
    num_clients = c2.slider("Edge Clients", 2, 6, 3)
    start_btn = c3.button("Start Federated Training 🚀", use_container_width=True)
    
    # Active Nodes UI
    nodes_html = f"<b>Active Edge Nodes:</b><br>" + "".join([f"<div class='pulse-node' title='Client {i+1}'></div>" for i in range(num_clients)])
    st.markdown(nodes_html, unsafe_allow_html=True)
    
    # Layout rendering areas before training starts so they dynamically fill!
    st.write("### Live Training Telemetry")
    progress_bar = st.progress(0)
    
    graph_col1, graph_col2, graph_col3 = st.columns(3)
    with graph_col1:
        st.caption("📈 Accuracy vs Rounds")
        acc_chart = st.empty()
        df_acc = pd.DataFrame(columns=["Round", "Accuracy"]).set_index("Round")
        acc_chart.line_chart(df_acc, color="#10b981", height=200)
    
    with graph_col2:
        st.caption("📉 Loss vs Rounds")
        loss_chart = st.empty()
        df_loss = pd.DataFrame(columns=["Round", "Loss"]).set_index("Round")
        loss_chart.line_chart(df_loss, color="#f43f5e", height=200)
        
    with graph_col3:
        if show_roc:
            st.caption("📊 ROC-AUC vs Rounds")
            roc_chart = st.empty()
            df_roc = pd.DataFrame(columns=["Round", "ROC-AUC"]).set_index("Round")
            roc_chart.line_chart(df_roc, color="#8b5cf6", height=200)
            
    st.caption("Live Training Action Logs")
    log_area = st.empty()
    
    if start_btn:
        logs_text = ["Federated Engine Booting Up...\n"]
        log_area.markdown(f'<div class="log-box">{logs_text[0]}</div>', unsafe_allow_html=True)
        
        hist_rounds, hist_acc, hist_loss, hist_roc = [], [], [], []
        
        def handle_live_callback(metrics, msg):
            if msg:
                logs_text[0] += f"> {msg}\n"
                log_area.markdown(f'<div class="log-box">{logs_text[0]}</div>', unsafe_allow_html=True)
                
            if metrics and "accuracy" in metrics:
                r = metrics["round"]
                # Update line charts
                hist_rounds.append(r)
                hist_acc.append(metrics["accuracy"])
                hist_loss.append(metrics["loss"])
                
                acc_chart.line_chart(pd.DataFrame({"Round": hist_rounds, "Accuracy": hist_acc}).set_index("Round"), color="#10b981", height=200)
                loss_chart.line_chart(pd.DataFrame({"Round": hist_rounds, "Loss": hist_loss}).set_index("Round"), color="#f43f5e", height=200)
                
                if show_roc and "roc_auc" in metrics:
                    hist_roc.append(metrics["roc_auc"])
                    roc_chart.line_chart(pd.DataFrame({"Round": hist_rounds, "ROC-AUC": hist_roc}).set_index("Round"), color="#8b5cf6", height=200)
                
                progress_bar.progress(r / num_rounds)
                
        # Block until completion with real-time Streamlit yielding
        final_res = sim_func(num_rounds, num_clients, live_callback=handle_live_callback)
        progress_bar.progress(1.0)
        st.success(f"{d_type} Protocol Accomplished!")
        
        cm = final_res["confusion_matrix"]
        m = final_res["metrics"]
        
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Final Accuracy", f"{m['Accuracy']:.3f}")
        s2.metric("Final F1", f"{m['F1-Score']:.3f}")
        if 'Precision' in m: s3.metric("Final Precision", f"{m['Precision']:.3f}")
        if cm: s4.metric("True Positives", cm['tp'])

elif nav == "🧪 Live Prediction":
    st.header(f"🧪 Live Prediction ({d_type})")
    
    if d_type == "Healthcare":
        from eda_mimic import CONTINUOUS_COLUMNS
        st.markdown("Enter clinical parameters to predict hospital mortality probability using the globally trained Random Forest.")
        with st.form("pred_form"):
            col1, col2 = st.columns(2)
            age = col1.number_input("Patient Age", 18, 100, 65)
            los = col1.number_input("ICU Length of Stay", 0.0, 100.0, 3.5)
            gender = col2.selectbox("Gender", ["M", "F"])
            adm_type = col2.selectbox("Admission Type", ["URGENT", "EW EMER.", "ELECTIVE", "DIRECT OBSERVATION"])
            careunit = st.selectbox("First Care Unit", ["MICU", "CVICU", "SICU", "TSICU", "CCU"])
            
            submit = st.form_submit_button("Run Probabilistic Inference")
            
        if submit:
            sample_df = pd.DataFrame(0, index=[0], columns=X_train.columns)
            cont_vals = pd.DataFrame([[age, los]], columns=CONTINUOUS_COLUMNS)
            scaled = scaler.transform(cont_vals)
            sample_df['anchor_age'] = scaled[0, 0]
            sample_df['los'] = scaled[0, 1]
            if f"gender_{gender}" in sample_df.columns: sample_df[f"gender_{gender}"] = 1
            if f"admission_type_{adm_type}" in sample_df.columns: sample_df[f"admission_type_{adm_type}"] = 1
            if f"first_careunit_Medical Intensive Care Unit ({careunit})" in sample_df.columns: 
                sample_df[f"first_careunit_Medical Intensive Care Unit ({careunit})"] = 1
                
            prob = results['Random Forest']['model_obj'].predict_proba(sample_df)[0][1]
            if prob > 0.4:
                st.error(f"🚨 **HIGH RISK**: {prob*100:.1f}% probability of non-survival.")
            else:
                st.success(f"✅ **LOWER RISK**: {prob*100:.1f}% probability of non-survival.")
    else:
        st.info(f"Custom form input mapping for {d_type} schema is under construction. Please use Centralized Models for analytical evaluations.")
