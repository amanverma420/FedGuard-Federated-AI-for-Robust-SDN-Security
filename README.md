# FedGuard: Privacy-Preserving Adversarially Robust IDS for SDN

## Project Structure
```
fedguard/
├── requirements.txt          # All dependencies
├── setup.py                  # Package setup
├── main.py                   # Entry point - runs full pipeline
├── config.py                 # Global configuration
│
├── data/
│   ├── data_loader.py        # Load CIC-IDS2018, NSL-KDD, UNSW-NB15
│   ├── preprocessor.py       # Feature engineering & normalization
│   └── synthetic_generator.py# Synthetic SDN traffic generator (when real data unavailable)
│
├── models/
│   ├── detector.py           # Deep Neural Network classifier
│   └── encoder.py            # Feature encoder
│
├── federated/
│   ├── client.py             # FL client (local SDN controller)
│   ├── server.py             # Byzantine-fault-tolerant aggregation server
│   └── aggregator.py         # FedAvg + Krum Byzantine defense
│
├── adversarial/
│   ├── gan.py                # GAN for adversarial attack generation
│   └── augmentor.py          # Adversarial training augmentation
│
├── dqn/
│   ├── environment.py        # SDN mitigation environment
│   ├── agent.py              # Deep Q-Network agent
│   └── replay_buffer.py      # Experience replay
│
├── simulation/
│   ├── sdn_simulator.py      # Simulated SDN topology
│   └── attack_simulator.py   # Attack traffic simulation
│
├── evaluation/
│   ├── metrics.py            # Accuracy, F1, latency metrics
│   └── benchmarks.py         # Full benchmark runner
│
├── utils/
│   ├── crypto.py             # Gradient encryption utilities
│   └── logger.py             # Logging utilities
│
└── dashboard/
    └── visualizer.py         # Training & results visualization
```

## Quick Start
```bash
pip install -r requirements.txt
python main.py
```

## Interactive GUI
For an interactive, user-friendly experience:
```bash
pip install -r requirements.txt
streamlit run app.py
# Or use the launcher:
python launch_gui.py
```
Or after installation:
```bash
pip install -e .
fedguard-gui
```

The GUI provides:
- 🔧 **Configuration Panel**: Adjust all parameters interactively
- 🚀 **Pipeline Runner**: Execute training with real-time progress
- 📊 **Results Dashboard**: View metrics, plots, and analysis
- 🎯 **Real-time Demo**: Simulate live traffic detection and mitigation

### GUI Features
- **Modern Dark Theme**: Cool, professional interface with gradient accents
- **Real-time Updates**: Live progress bars and status updates during training
- **Interactive Plots**: Matplotlib integration for training curves and analysis
- **Parameter Tuning**: Sliders and inputs for all configuration options
- **Live Demo**: Real-time attack simulation with detection and mitigation
- **Results Analysis**: Confusion matrices, performance metrics, and insights
- **Responsive Design**: Works on desktop and mobile browsers

## Pipeline
1. Data loading & preprocessing (synthetic if datasets unavailable)
2. Federated Learning across 5 simulated SDN controllers
3. GAN adversarial augmentation
4. DQN autonomous mitigation training
5. Full evaluation & benchmarking
6. Results visualization
