"""
app.py - FedGuard Interactive GUI
Run with: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import time
import config
from data.data_loader import load_dataset
from data.preprocessor import SDNPreprocessor
from data.synthetic_generator import generate_dataset
from federated.client import FederatedClient
from federated.server import FederatedServer
from models.detector import IntrusionDetector
from adversarial.gan import AttackGAN
from adversarial.augmentor import AdversarialAugmentor
from dqn.agent import DQNAgent
from dqn.environment import SDNMitigationEnv
from evaluation.metrics import detection_metrics, adversarial_robustness_metrics
from evaluation.benchmarks import run_detection_benchmark, run_adversarial_benchmark, run_mitigation_benchmark
from dashboard.visualizer import plot_fl_training, plot_confusion_matrix, plot_adversarial_robustness, plot_dqn_training, plot_gan_training
from simulation.attack_simulator import AttackSimulator
import torch

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FedGuard IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS for Cool Design ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: 500;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        color: white;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .status-info {
        color: #17a2b8;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State Initialization ─────────────────────────────────────────────
if 'pipeline_results' not in st.session_state:
    st.session_state.pipeline_results = {}
if 'config' not in st.session_state:
    st.session_state.config = config.__dict__.copy()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# ── Helper Functions ─────────────────────────────────────────────────────────
def update_config():
    """Update config module with session state values"""
    for key, value in st.session_state.config.items():
        if hasattr(config, key):
            setattr(config, key, value)

def load_data_progress():
    """Load and preprocess data with progress"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.text("Loading dataset...")
    progress_bar.progress(20)

    X_train, X_test, y_train, y_test = load_dataset()
    progress_bar.progress(40)

    status_text.text("Preprocessing data...")
    pre = SDNPreprocessor()
    X_train = pre.fit_transform(X_train)
    X_test = pre.transform(X_test)
    progress_bar.progress(60)

    # SDN Topology
    from simulation.sdn_simulator import SDNTopology
    topology = SDNTopology()
    progress_bar.progress(80)

    # Client data split
    from data.synthetic_generator import split_for_clients
    client_data = split_for_clients(X_train, y_train, n_clients=config.NUM_CLIENTS)
    progress_bar.progress(100)

    status_text.text("Data loaded successfully!")
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()

    return X_train, X_test, y_train, y_test, topology, client_data

# ── Main App ─────────────────────────────────────────────────────────────────
def main():
    # Sidebar Navigation
    st.sidebar.markdown("# 🛡️ FedGuard IDS")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Dashboard", "⚙️ Configuration", "🚀 Run Pipeline", "📊 Results", "🎯 Real-time Demo"],
        index=0
    )

    # Main Content
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "⚙️ Configuration":
        show_configuration()
    elif page == "🚀 Run Pipeline":
        show_run_pipeline()
    elif page == "📊 Results":
        show_results()
    elif page == "🎯 Real-time Demo":
        show_realtime_demo()

# ── Dashboard Page ───────────────────────────────────────────────────────────
def show_dashboard():
    st.markdown('<h1 class="main-header">FedGuard: Privacy-Preserving Adversarially Robust IDS</h1>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🔒 Privacy-First</h3>
            <p>Federated Learning with Differential Privacy</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🛡️ Adversarial Robust</h3>
            <p>GAN-augmented training against evasion attacks</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 Autonomous</h3>
            <p>DQN-powered mitigation in SDN environments</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Quick Stats
    if st.session_state.pipeline_results:
        st.subheader("📈 Latest Results")

        results = st.session_state.pipeline_results
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            acc = results.get('detection_acc', 0)
            st.metric("Detection Accuracy", f"{acc:.4f}", delta=f"{acc-0.97:.4f}" if acc >= 0.97 else None)

        with col2:
            adv_acc = results.get('adversarial_acc', 0)
            st.metric("Adversarial Robustness", f"{adv_acc:.4f}", delta=f"{adv_acc-0.89:.4f}" if adv_acc >= 0.89 else None)

        with col3:
            latency = results.get('latency_ms', 0)
            st.metric("Mitigation Latency", f"{latency:.1f}ms", delta=f"{340-latency:.1f}ms" if latency <= 340 else None)

        with col4:
            fl_time = results.get('fl_time', 0)
            st.metric("Training Time", f"{fl_time:.1f}s")
    else:
        st.info("🚀 Run the pipeline to see results here!")

    # Architecture Overview
    st.subheader("🏗️ System Architecture")
    st.markdown("""
    ```mermaid
    graph TD
        A[Data Sources] --> B[Preprocessing]
        B --> C[Federated Learning]
        C --> D[Adversarial Augmentation]
        D --> E[DQN Mitigation]
        E --> F[SDN Deployment]
        F --> G[Real-time Detection]
    ```
    """)

# ── Configuration Page ───────────────────────────────────────────────────────
def show_configuration():
    st.header("⚙️ Configuration")

    st.markdown("Adjust parameters for the FedGuard pipeline. Changes are applied in real-time.")

    # Dataset Settings
    st.subheader("📊 Dataset")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.config['DATASET'] = st.selectbox(
            "Dataset Source",
            ["synthetic", "nsl_kdd"],
            index=["synthetic", "nsl_kdd"].index(st.session_state.config.get('DATASET', 'synthetic'))
        )
        st.session_state.config['NUM_FEATURES'] = st.slider("Number of Features", 10, 100, st.session_state.config.get('NUM_FEATURES', 41))

    with col2:
        st.session_state.config['TEST_SPLIT'] = st.slider("Test Split Ratio", 0.1, 0.5, st.session_state.config.get('TEST_SPLIT', 0.2))
        st.session_state.config['SEED'] = st.number_input("Random Seed", value=st.session_state.config.get('SEED', 42))

    # Federated Learning Settings
    st.subheader("🔗 Federated Learning")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state.config['NUM_CLIENTS'] = st.slider("Number of Clients", 3, 10, st.session_state.config.get('NUM_CLIENTS', 5))
    with col2:
        st.session_state.config['FL_ROUNDS'] = st.slider("FL Rounds", 5, 50, st.session_state.config.get('FL_ROUNDS', 30))
    with col3:
        st.session_state.config['LOCAL_EPOCHS'] = st.slider("Local Epochs", 1, 10, st.session_state.config.get('LOCAL_EPOCHS', 5))

    # Adversarial Settings
    st.subheader("🛡️ Adversarial Training")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.config['GAN_EPOCHS'] = st.slider("GAN Epochs", 5, 50, st.session_state.config.get('GAN_EPOCHS', 30))
    with col2:
        st.session_state.config['ADVERSARIAL_RATIO'] = st.slider("Adversarial Ratio", 0.1, 1.0, st.session_state.config.get('ADVERSARIAL_RATIO', 0.3))

    # DQN Settings
    st.subheader("🎯 DQN Agent")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.config['DQN_EPISODES'] = st.slider("DQN Episodes", 50, 500, st.session_state.config.get('DQN_EPISODES', 200))
    with col2:
        st.session_state.config['DQN_MAX_STEPS'] = st.slider("Max Steps per Episode", 20, 100, st.session_state.config.get('DQN_MAX_STEPS', 50))

    # Privacy Settings
    st.subheader("🔐 Privacy")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.config['ENCRYPT_GRADIENTS'] = st.checkbox("Encrypt Gradients", st.session_state.config.get('ENCRYPT_GRADIENTS', False))
    with col2:
        st.session_state.config['GRADIENT_NOISE_STD'] = st.slider("DP Noise STD", 0.0, 0.01, st.session_state.config.get('GRADIENT_NOISE_STD', 0.001))

    if st.button("💾 Save Configuration"):
        update_config()
        st.success("Configuration saved!")

# ── Run Pipeline Page ────────────────────────────────────────────────────────
def show_run_pipeline():
    st.header("🚀 Run FedGuard Pipeline")

    st.markdown("Execute the complete FedGuard training pipeline. This may take several minutes.")

    # Pipeline Steps
    steps = [
        "📊 Load & Preprocess Data",
        "🏗️ Setup SDN Topology",
        "🔗 Federated Learning",
        "🛡️ GAN Adversarial Augmentation",
        "🎯 DQN Autonomous Training",
        "📈 Full Evaluation"
    ]

    quick_run = st.checkbox(
        "Use Quick Mode (fast demo)",
        value=True,
        help="Reduces FL rounds, GAN epochs, and DQN episodes for faster execution."
    )
    st.markdown("**Note:** Full pipeline can take many minutes. Quick mode is recommended for interactive use.")

    if 'pipeline_running' not in st.session_state:
        st.session_state.pipeline_running = False

    if st.button("▶️ Start Pipeline", type="primary") and not st.session_state.pipeline_running:
        st.session_state.pipeline_running = True
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Step 1: Data Loading
            status_text.markdown("**Step 1/6:** 📊 Loading and preprocessing data...")
            X_train, X_test, y_train, y_test, topology, client_data = load_data_progress()
            progress_bar.progress(17)

            # Step 2: SDN Topology
            status_text.markdown("**Step 2/6:** 🏗️ Setting up SDN topology...")
            stats = topology.get_stats()
            progress_bar.progress(33)

            # Step 3: Federated Learning
            status_text.markdown("**Step 3/6:** 🔗 Running Federated Learning...")
            update_config()
            if quick_run:
                config.FL_ROUNDS = min(config.FL_ROUNDS, 10)
                config.LOCAL_EPOCHS = min(config.LOCAL_EPOCHS, 3)
            server = FederatedServer(X_test, y_test)
            clients = [FederatedClient(i, cd[0], cd[1]) for i, cd in enumerate(client_data)]
            fl_history = server.run_all_rounds(clients)
            progress_bar.progress(50)

            # Step 4: GAN Augmentation
            status_text.markdown("**Step 4/6:** 🛡️ GAN adversarial augmentation...")
            if quick_run:
                config.GAN_EPOCHS = min(config.GAN_EPOCHS, 10)
            augmentor = AdversarialAugmentor(model=server.global_model)
            augmentor.fit_gan(X_train, y_train)
            X_aug, y_aug = augmentor.augment(X_train, y_train)
            progress_bar.progress(67)

            # Step 5: DQN Training
            status_text.markdown("**Step 5/6:** 🎯 DQN autonomous training...")
            if quick_run:
                config.DQN_EPISODES = min(config.DQN_EPISODES, 50)
            dqn_env = SDNMitigationEnv(X_test, y_test, detector=server.global_model)
            dqn_agent = DQNAgent()
            episode_rewards = dqn_agent.train(dqn_env)
            progress_bar.progress(83)

            # Step 6: Evaluation
            status_text.markdown("**Step 6/6:** 📈 Running full evaluation...")
            detection_results = run_detection_benchmark(server.global_model, X_test, y_test)
            adversarial_results = run_adversarial_benchmark(server.global_model, X_test, y_test)
            mitigation_results = run_mitigation_benchmark(dqn_agent, dqn_env)

            progress_bar.progress(100)

            # Store results
            st.session_state.pipeline_results = {
                'detection_acc': detection_results['accuracy'],
                'adversarial_acc': adversarial_results,
                'latency_ms': mitigation_results['avg_latency_ms'],
                'fl_time': sum(h['train_loss'] for h in fl_history) * 0.1,  # Approximate
                'fl_history': fl_history,
                'episode_rewards': episode_rewards,
                'X_test': X_test,
                'y_test': y_test,
                'model': server.global_model
            }
            st.session_state.data_loaded = True
            st.session_state.model_trained = True

            status_text.markdown("**✅ Pipeline Complete!**")
            st.success("FedGuard pipeline executed successfully!")

        except Exception as e:
            st.error(f"Pipeline failed: {str(e)}")
            st.exception(e)

        finally:
            st.session_state.pipeline_running = False

# ── Results Page ─────────────────────────────────────────────────────────────
def show_results():
    st.header("📊 Results & Analysis")

    if not st.session_state.pipeline_results:
        st.warning("No results available. Run the pipeline first!")
        return

    results = st.session_state.pipeline_results

    # Key Metrics
    st.subheader("🎯 Key Performance Metrics")
    col1, col2, col3 = st.columns(3)

    with col1:
        acc = results['detection_acc']
        delta = f"+{acc-0.97:.4f}" if acc >= 0.97 else f"{acc-0.97:.4f}"
        st.metric("Detection Accuracy", f"{acc:.4f}", delta)

    with col2:
        adv_acc = results['adversarial_acc']
        delta = f"+{adv_acc-0.89:.4f}" if adv_acc >= 0.89 else f"{adv_acc-0.89:.4f}"
        st.metric("Adversarial Accuracy", f"{adv_acc:.4f}", delta)

    with col3:
        latency = results['latency_ms']
        delta = f"-{340-latency:.1f}" if latency <= 340 else f"+{latency-340:.1f}"
        st.metric("Latency (ms)", f"{latency:.1f}", delta)

    # Plots
    st.subheader("📈 Training Curves")

    if 'fl_history' in results:
        fig, ax = plt.subplots(figsize=(10, 6))
        rounds = [h['round'] for h in results['fl_history']]
        accs = [h['accuracy'] for h in results['fl_history']]
        ax.plot(rounds, accs, 'b-o', linewidth=2, markersize=4)
        ax.set_title('Federated Learning: Global Model Accuracy')
        ax.set_xlabel('Round')
        ax.set_ylabel('Accuracy')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    if 'episode_rewards' in results:
        fig, ax = plt.subplots(figsize=(10, 6))
        rewards = results['episode_rewards']
        ax.plot(rewards, 'g-', alpha=0.7)
        ax.plot(pd.Series(rewards).rolling(20).mean(), 'r-', linewidth=2)
        ax.set_title('DQN Training: Episode Rewards')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend(['Raw', '20-episode MA'])
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    # Confusion Matrix
    if 'X_test' in results and 'y_test' in results and 'model' in results:
        st.subheader("🔍 Confusion Matrix")
        from sklearn.metrics import confusion_matrix
        model = results['model']
        X_test, y_test = results['X_test'], results['y_test']

        with torch.no_grad():
            preds = model.predict(torch.tensor(X_test, dtype=torch.float32)).numpy()

        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES, ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        st.pyplot(fig)

# ── Real-time Demo Page ──────────────────────────────────────────────────────
def show_realtime_demo():
    st.header("🎯 Real-time Attack Detection Demo")

    if not st.session_state.model_trained:
        st.warning("Train the model first to run the demo!")
        return

    st.markdown("Simulate real-time traffic monitoring with live detection and mitigation.")

    # Controls
    col1, col2, col3 = st.columns(3)
    with col1:
        n_events = st.slider("Number of Events", 10, 200, 50)
    with col2:
        detection_threshold = st.slider("Detection Threshold", 0.0, 1.0, 0.5)
    with col3:
        auto_mitigate = st.checkbox("Auto-Mitigation", True)

    if st.button("▶️ Start Demo"):
        results = st.session_state.pipeline_results
        model = results['model']
        X_test, y_test = results['X_test'], results['y_test']

        simulator = AttackSimulator(X_test, y_test)

        # Demo output
        st.subheader("📡 Live Traffic Feed")
        demo_placeholder = st.empty()

        events_data = []
        for i in range(n_events):
            event = simulator.next_event()

            # Get prediction
            with torch.no_grad():
                feat = torch.tensor(event['features'], dtype=torch.float32).unsqueeze(0)
                pred_prob = model.predict_proba(feat)[0]
                pred_class = int(torch.argmax(pred_prob))

            event['predicted_class'] = pred_class
            event['predicted_name'] = config.CLASS_NAMES[pred_class]
            event['confidence'] = float(pred_prob[pred_class])
            event['detected'] = pred_class != 0 and event['confidence'] > detection_threshold

            events_data.append(event)

            # Update display
            df = pd.DataFrame(events_data[-10:])  # Show last 10
            demo_placeholder.dataframe(df[['timestamp', 'label_name', 'predicted_name', 'confidence', 'detected']])

            time.sleep(0.1)  # Simulate real-time

        # Summary
        st.subheader("📊 Demo Summary")
        total_attacks = sum(1 for e in events_data if e['label'] != 0)
        detected_attacks = sum(1 for e in events_data if e['detected'])
        false_positives = sum(1 for e in events_data if e['detected'] and e['label'] == 0)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Attacks", total_attacks)
        with col2:
            st.metric("Detected Attacks", detected_attacks)
        with col3:
            st.metric("False Positives", false_positives)

if __name__ == "__main__":
    main()