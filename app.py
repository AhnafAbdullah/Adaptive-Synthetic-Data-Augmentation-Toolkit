import streamlit as st
import pandas as pd
import numpy as np
import time
import io
import base64

from sklearn.model_selection import train_test_split
from data_utils import  preprocess_features
from models_utils import build_and_train_model, evaluate_performance
from generator_utils import train_generator
from loop_engine import execute_adaptive_loop_stream
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Adaptive Synthetic Data Engine", page_icon="🧬", layout="wide")
st.markdown("""
<style>
    :root {
        --primary-color: #6C5CE7;
        --secondary-color: #00CEC9;
        --bg-color: #0F0F1A;
        --card-bg: #1A1A2E;
        --text-color: #DFDFDF;
    }
    
    .stApp {
        background-color: var(--bg-color) !important;
        color: var(--text-color);
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stSidebar"] {
        background-color: var(--card-bg) !important;
        border-right: 1px solid #2D2D44 !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(to right, #00CEC9, #6C5CE7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    h1, h2, h3, h4, h5, h6 {
        color: white !important;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    .stButton > button, [data-testid="stFileUploader"] button {
        background-color: rgba(28, 131, 225, 0.1) !important;
        color: white !important;
        border: 1px solid rgba(28, 131, 225, 0.5) !important;
    }
    
    .stButton > button:hover, [data-testid="stFileUploader"] button:hover {
        background-color: rgba(28, 131, 225, 0.25) !important;
        border-color: rgba(28, 131, 225, 1.0) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

def extract_weak_cohorts(X_test, y_test, y_pred, target_col):
    """Identify weak subsets to target for synthetic generation"""
    errors = (y_test != y_pred).astype(int)
    if errors.sum() == 0: return []
    
    fn_mask = (y_test == 1) & (y_pred == 0)
    cohorts = []
    
    if fn_mask.sum() > 0:
        fn_data = X_test[fn_mask]
        cat_cols = [c for c in X_test.columns if X_test[c].nunique() < 15]
        
        if cat_cols:
            for feat in cat_cols[:2]:
                val = fn_data[feat].mode()[0]
                cohorts.append({"name": f"False Negatives (where {feat}={val})", "conditions": {feat: val}, "label": 1})
        else:
            cohorts.append({"name": "Minority Class Misclassifications", "conditions": {}, "label": 1})
            
    if not cohorts:
        cohorts.append({"name": "General Minority Boosting", "conditions": {}, "label": 1})
        
    return cohorts


def main():
    st.title("🧬 Adaptive Synthetic Data Augmentation Toolkit")
    st.write("A closed-loop engine that automatically detects weak model cohorts and generates targeted synthetic data to boost performance.")
    
    # Initialize session state
    for k in ['df', 'target_col', 'X_train', 'X_test', 'y_train', 'y_test', 
              'X_train_proc', 'X_test_proc', 'preprocessor', 'baseline_model', 
              'baseline_metrics', 'cohorts', 'generator', 'loop_history', 'abort', 'final_model']:
        if k not in st.session_state:
            st.session_state[k] = None

    # Sidebar: Config & Data
    with st.sidebar:
        st.header("1. Configuration")
        uploaded_file = st.file_uploader("Upload CSV Dataset", type="csv")
        
        if uploaded_file is not None:
            # Check if this is a newly uploaded file
            if st.session_state.get("last_uploaded") != uploaded_file.name:
                with st.spinner("Loading dataset..."):
                    df = pd.read_csv(uploaded_file)
                    # Auto-fill missing values
                    for col in df.columns:
                        if df[col].dtype in [np.float64, np.int64]:
                            df[col] = df[col].fillna(df[col].median())
                        else:
                            df[col] = df[col].fillna(df[col].mode()[0])
                    st.session_state.df = df
                    st.session_state.last_uploaded = uploaded_file.name
                    # Clear run state for new files
                    st.session_state.baseline_model = None
                    st.session_state.cohorts = None

        if st.session_state.df is not None:
            df = st.session_state.df
            
            # Target Column Selection
            possible_targets = [c for c in df.columns if df[c].nunique() == 2]
            if not possible_targets: 
                possible_targets = df.columns.tolist()
                
            target_col = st.selectbox("Select Target Column", possible_targets)
            model_type = st.selectbox("Model Architecture", ["random_forest", "logistic_regression", "decision_tree"])
            
            st.markdown("---")
            st.header("Loop Parameters")
            max_iter = st.slider("Max Iterations", 1, 10, 5)
            n_samples = st.number_input("Samples Per Cohort/Iter", min_value=10, max_value=5000, value=100)
            threshold = st.number_input("Early Stop ΔAUC Threshold", value=0.001, format="%.4f")
            
            if st.button("Initialize & Train Baseline"):
                pb = st.progress(0, text="Preparing data...")
                
                # Process & Split
                df[target_col] = df[target_col].astype(int)
                X = df.drop(columns=[target_col])
                y = df[target_col]
                                
                pb.progress(15, text="Splitting train and test sets...")
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                
                # Store Raw
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.target_col = target_col
                
                pb.progress(35, text="Preprocessing features...")
                # Preprocess
                X_train_proc, X_test_proc, preprocessor, _ = preprocess_features(X_train, X_test)
                st.session_state.X_train_proc = X_train_proc
                st.session_state.X_test_proc = X_test_proc
                st.session_state.preprocessor = preprocessor
                
                pb.progress(60, text="Training baseline model...")
                # Baseline Model
                model = build_and_train_model(X_train_proc, y_train.values, model_type=model_type, random_state=42)
                st.session_state.baseline_model = model
                
                pb.progress(85, text="Evaluating performance & identifying error cohorts...")
                # Evaluate
                metrics = evaluate_performance(model, X_test_proc, y_test.values)
                st.session_state.baseline_metrics = metrics
                
                # Identify Cohorts
                y_pred = model.predict(X_test_proc)
                st.session_state.cohorts = extract_weak_cohorts(X_test, y_test.values, y_pred, target_col)
                
                pb.progress(100, text="Baseline training complete!")
                time.sleep(1)
                pb.empty()
                st.success("Baseline training complete!")

    # Main dashboard
    if st.session_state.df is None:
        st.info("👈 Please upload a dataset in the sidebar to begin.")
        return

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Data Overview")
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
    with col2:
        if st.session_state.baseline_metrics:
            st.subheader("Baseline Performance")
            m1, m2, m3 = st.columns(3)
            m1.metric("AUC Score", f"{st.session_state.baseline_metrics['roc_auc']:.4f}")
            m2.metric("F1-Score", f"{st.session_state.baseline_metrics['f1']:.4f}")
            m3.metric("Recall", f"{st.session_state.baseline_metrics['recall']:.4f}")
            m4, m5 = st.columns(2)
            m4.metric("Accuracy", f"{st.session_state.baseline_metrics['accuracy']:.4f}")
            m5.metric("Precision", f"{st.session_state.baseline_metrics['precision']:.4f}")

    if st.session_state.cohorts:
        st.markdown("---")
        st.subheader("🔍 Error Analysis & Targeted Cohorts")
        st.write("The baseline model struggled with the following cohorts. The generator will focus on synthesizing these specific scenarios.")
        st.table(pd.DataFrame(st.session_state.cohorts)[['name', 'label']])

    if st.session_state.baseline_model is not None:
        st.markdown("---")
        st.subheader("🚀 Adaptive Augmentation Loop")
        
        col_btn1, col_btn2 = st.columns([2, 1])
        with col_btn1:
            start_loop = st.button("Run Optimization Loop")
        with col_btn2:
            abort_btn = st.button("Stop Run (Abort)")
            if abort_btn:
                st.session_state.abort = True
                st.experimental_rerun()
                
        if start_loop:
            st.session_state.abort = False
            
            # Stage 1: Train Synthesizer
            if st.session_state.generator is None:
                pb_synth = st.progress(0, text="Formatting data for SDV Synthesizer...")
                real_data = pd.concat([st.session_state.X_train, st.session_state.y_train.rename(st.session_state.target_col)], axis=1)
                
                pb_synth.progress(50, text="Fitting SDV Synthesizer (this might take a moment)...")
                generator = train_generator(real_data, target_col=st.session_state.target_col, generator_type='copula')
                st.session_state.generator = generator
                
                pb_synth.progress(100, text="Synthesizer ready!")
                time.sleep(1)
                pb_synth.empty()
            
            st.write("Starting iterations...")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            chart_placeholder = st.empty()
            
            loop_gen = execute_adaptive_loop_stream(
                st.session_state.X_train, st.session_state.y_train,
                st.session_state.X_test, st.session_state.y_test,
                st.session_state.X_train_proc, st.session_state.X_test_proc,
                st.session_state.preprocessor, st.session_state.baseline_model,
                st.session_state.generator, st.session_state.cohorts,
                target_col=st.session_state.target_col,
                model_type=model_type,
                max_iterations=max_iter,
                n_samples_per_iter=n_samples,
                early_stop_threshold=threshold,
                random_state=42
            )
            
            metrics_acc = []
            
            for state in loop_gen:
                if st.session_state.abort:
                    status_text.error("Run aborted by user!")
                    break
                    
                if state["status"] == "baseline":
                    metrics_acc = state["metrics_history"]
                    
                elif state["status"] == "generating":
                    pct = int(((state["iteration"] - 0.5) / state["max_iterations"]) * 100)
                    progress_bar.progress(pct)
                    status_text.info(f"Iteration {state['iteration']}/{state['max_iterations']}: Sample generation...")
                    
                elif state["status"] == "training":
                    status_text.info(f"Iteration {state['iteration']}/{state['max_iterations']}: Retraining model...")
                    
                elif state["status"] == "iter_complete":
                    pct = int((state["iteration"] / max_iter) * 100)
                    progress_bar.progress(pct)
                    metrics_acc = state["metrics_history"]
                    
                    # Update chart
                    df_chart = pd.DataFrame(metrics_acc)[['iteration', 'roc_auc', 'f1']].set_index('iteration')
                    chart_placeholder.line_chart(df_chart)
                    
                elif state["status"] == "early_stop":
                    status_text.warning(f"Early Stopping Triggered: {state.get('reason', '')}")
                    
                elif state["status"] == "complete":
                    status_text.success("Optimization Loop Complete!")
                    st.session_state.final_model = state["best_model"]
                    
                    st.markdown("### Final Comparison")
                    base_m = metrics_acc[0]
                    final_m = metrics_acc[-1]
                    
                    fm1, fm2, fm3 = st.columns(3)
                    fm1.metric("AUC Score", f"{final_m['roc_auc']:.4f}", f"{(final_m['roc_auc'] - base_m['roc_auc']):.4f}")
                    fm2.metric("F1-Score", f"{final_m['f1']:.4f}", f"{(final_m['f1'] - base_m['f1']):.4f}")
                    fm3.metric("Recall", f"{final_m['recall']:.4f}", f"{(final_m['recall'] - base_m['recall']):.4f}")
                    
                    fm4, fm5 = st.columns(2)
                    fm4.metric("Accuracy", f"{final_m['accuracy']:.4f}", f"{(final_m['accuracy'] - base_m['accuracy']):.4f}")
                    fm5.metric("Precision", f"{final_m['precision']:.4f}", f"{(final_m['precision'] - base_m['precision']):.4f}")
                    
            if not st.session_state.abort:
                st.balloons()

if __name__ == "__main__":
    main()
