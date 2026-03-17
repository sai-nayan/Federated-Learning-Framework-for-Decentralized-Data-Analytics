import streamlit as st
import pandas as pd
import time
import importlib
import numpy as np
import altair as alt

from federated_nn import run_simulation as run_nn_simulation
from federated_student import run_simulation as run_student_simulation
from federated_finance import run_simulation as run_finance_simulation

# ==========================================
# PAGE CONFIG AND STYLING
# ==========================================
st.set_page_config(
    page_title="Federated Learning Hub",
    page_icon="🤖",
    layout="wide",
)

# Domain colors mapping
DOMAIN_THEMES = {
    "Healthcare (NN)": "#10b981", # Emerald
    "Finance (XGBoost)": "#06b6d4", # Cyan
    "Education (Random Forest)": "#8b5cf6" # Violet
}

# Apply global custom styling
st.markdown("""
<style>
    /* Card styling */
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
    
    /* Pulsing dots animation for clients */
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 0, 0, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(0, 0, 0, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 0, 0, 0); }
    }
    .node-container {
        display: flex;
        justify-content: center;
        gap: 15px;
        margin: 20px 0;
        flex-wrap: wrap;
    }
    .client-node {
        width: 25px;
        height: 25px;
        border-radius: 50%;
        display: inline-block;
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("FL Configuration")
st.sidebar.markdown("Configure the federated learning parameters for the selected domain.")

domain = st.sidebar.radio(
    "Select Domain",
    options=["Healthcare (NN)", "Finance (XGBoost)", "Education (Random Forest)"],
)

theme_color = DOMAIN_THEMES[domain]

st.sidebar.markdown("---")
num_rounds = st.sidebar.slider("Communication Rounds", min_value=1, max_value=20, value=5)
num_clients = st.sidebar.slider("Number of Clients", min_value=2, max_value=10, value=3)

# ==========================================
# MAIN DASHBOARD CONTENT
# ==========================================

st.title("Federated Learning Real-Time Dashboard")
st.markdown(f"**Current Domain:** <span style='color:{theme_color}; font-weight:bold;'>{domain}</span>", unsafe_allow_html=True)

# Generate pulsing client nodes based on selected number of clients
st.markdown("### Participating Clients")
nodes_html = "<div class='node-container'>"
for i in range(num_clients):
    delay = i * 0.2  # offset animations
    nodes_html += f"<div class='client-node' style='background-color: {theme_color}; animation-delay: {delay}s;' title='Client {i+1}'></div>"
nodes_html += "</div>"
st.markdown(nodes_html, unsafe_allow_html=True)

st.markdown("---")

run_button = st.button("Start Federated Training 🔥")

# Helper function to create animated charts
def draw_animated_chart(history_df, metric_col, title, color):
    chart = alt.Chart(history_df).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X('rounds:Q', title="Round"),
        y=alt.Y(f'{metric_col}:Q', title=title, scale=alt.Scale(zero=False)),
        color=alt.value(color),
        tooltip=['rounds', metric_col]
    ).properties(
        height=300,
        width='container'
    ).interactive()
    return chart

if run_button:
    # Set up placeholders for live updates
    st.subheader("Training Progress")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    col_chart_acc, col_chart_loss = st.columns(2)
    with col_chart_acc:
        acc_chart_placeholder = st.empty()
    with col_chart_loss:
        loss_chart_placeholder = st.empty()
        
    global_cm_placeholder = st.empty()
    final_metrics_placeholder = st.empty()

    # Determine which simulation to run
    if domain == "Healthcare (NN)":
        sim_func = run_nn_simulation
    elif domain == "Finance (XGBoost)":
        sim_func = run_finance_simulation
    else:
        sim_func = run_student_simulation

    # Display initial empty charts
    empty_df = pd.DataFrame({"rounds": [], "accuracy": [], "loss": []})
    acc_chart_placeholder.altair_chart(draw_animated_chart(empty_df, "accuracy", "Accuracy", theme_color), use_container_width=True)
    loss_chart_placeholder.altair_chart(draw_animated_chart(empty_df, "loss", "Loss", "#f43f5e"), use_container_width=True)

    # Note: Streamlit doesn't natively stream from Flower easily without custom callbacks.
    # We will run the simulation and then "replay" the results for the live animation effect.
    with st.spinner(f"Running Flower Server for {domain}..."):
        try:
            results = sim_func(num_rounds, num_clients)
            success = True
        except Exception as e:
            import traceback
            traceback.print_exc()
            st.error(f"Simulation failed: {e}")
            success = False
            
    if success:
        history = results.get("history", {"rounds": [], "accuracy": [], "loss": []})
        
        # Live Animation Loop
        st.toast("Simulation finished! Replaying metrics...", icon="📈")
        for i in range(len(history["rounds"])):
            time.sleep(0.5) # Fake delay for live effect
            
            current_round = i + 1
            progress_bar.progress(current_round / num_rounds)
            status_text.text(f"Processing Round {current_round}/{num_rounds}...")
            
            # Slice data up to current round
            current_df = pd.DataFrame({
                "rounds": history["rounds"][:current_round],
                "accuracy": history["accuracy"][:current_round],
                "loss": history["loss"][:current_round]
            })
            
            # Update charts
            acc_chart_placeholder.altair_chart(draw_animated_chart(current_df, "accuracy", "Global Accuracy", theme_color), use_container_width=True)
            loss_chart_placeholder.altair_chart(draw_animated_chart(current_df, "loss", "Global Loss", "#f43f5e"), use_container_width=True)
        
        status_text.text("Training Complete!")
        
        # Display Final Metrics
        st.markdown("---")
        st.subheader("Global Model Evaluation")
        
        metrics = results.get("metrics", {})
        cols = st.columns(len(metrics))
        
        for idx, (key, value) in enumerate(metrics.items()):
            color = theme_color if idx == 0 else "#8b5cf6"
            with cols[idx]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{key}</div>
                    <div class="metric-value" style="color: {color}">{value:.4f}</div>
                </div>
                """, unsafe_allow_html=True)
                
        # Display Confusion Matrix
        cm = results.get("confusion_matrix", {})
        if cm and any(cm.values()):
            st.markdown("### Global Confusion Matrix")
            # Create a stylized table using columns
            c1, c2, c3 = st.columns([1, 2, 2])
            with c1:
                st.write("")
                st.write("**Actual 0**")
                st.write("**Actual 1**")
            with c2:
                st.write("**Predicted 0**")
                st.info(f"True Negative: {cm.get('tn', 0)}")
                st.error(f"False Negative: {cm.get('fn', 0)}")
            with c3:
                st.write("**Predicted 1**")
                st.warning(f"False Positive: {cm.get('fp', 0)}")
                st.success(f"True Positive: {cm.get('tp', 0)}")
