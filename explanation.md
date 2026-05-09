
---

## What is FedGuard and why does it exist?

Traditional intrusion detection systems (IDS) sit in one place and see all the network traffic. That's fine for a single organization, but **Software-Defined Networks (SDNs)** spread across many controllers — hospitals, banks, universities — and you can't just ship everyone's raw traffic data to one central server. That would be a massive privacy violation.

FedGuard solves this with four technologies layered on top of each other: Federated Learning (privacy), GAN augmentation (robustness), Deep Q-Network (automated response), and a Streamlit dashboard (visibility).

Let's start with the big picture:---

## Step 1 — The dataset (`data/`)

### Theory
Network traffic has 41 measurable features per connection: how long it lasted, how many bytes moved, how many failed logins happened, what protocol was used, etc. This is the NSL-KDD schema. Each sample gets one of five labels:

- **Normal** — legitimate traffic
- **DoS** — flooding attacks (SYN flood, Smurf)
- **Probe** — port scanning, reconnaissance
- **R2L** — remote-to-local login attempts
- **U2R** — privilege escalation once inside

FedGuard generates 50,000 synthetic samples (`data/synthetic_generator.py`) using statistical profiles for each class — each class has a characteristic mean and standard deviation for each feature. DoS traffic has very high `count` (400+) and near-zero `duration` (0.1s), while U2R has high `root_shell` and many `num_compromised` entries.

### What `config.py` is doing
`config.py` is the single source of truth for every number in the system. Instead of scattering `5` and `30` and `0.001` across files, one file holds `NUM_CLIENTS = 5`, `FL_ROUNDS = 30`, `GRADIENT_NOISE_STD = 0.001`. Every other file imports this. This is a classic software engineering pattern called a **configuration singleton**.

### What preprocessing does
`SDNPreprocessor` does two things. First, it **log-transforms** skewed features — if `src_bytes` ranges from 0 to 5,000,000, the distribution is wildly skewed. Taking `log(1 + x)` compresses the tail so the model can learn from it. Second, it **standardizes** with `StandardScaler` — subtracts the mean and divides by the standard deviation so every feature has mean=0 and std=1. Neural networks train much better on standardized data.

Critically, `fit()` is called only on training data, and `transform()` uses those saved statistics on test data. This prevents **data leakage** — the model never sees statistics from the future.

---

## Step 2 — The detection model (`models/detector.py`)

### Theory
`IntrusionDetector` is a **feedforward neural network** (also called a multilayer perceptron). It takes 41 input features and outputs a probability over 5 classes.

Here's what happens inside on each forward pass:

```
Input (41 features)
  → Linear(41 → 256) → BatchNorm → ReLU → Dropout(0.3)
  → Linear(256 → 128) → BatchNorm → ReLU → Dropout(0.3)
  → Linear(128 → 64) → BatchNorm → ReLU → Dropout(0.3)
  → Linear(64 → 5)   ← raw logits, one per class
```

- **BatchNorm** normalizes the activations within each mini-batch, stabilizing training
- **ReLU** is the activation function: `max(0, x)` — it introduces nonlinearity so the model can learn complex patterns
- **Dropout(0.3)** randomly zeros 30% of neurons during training, preventing overfitting
- The final layer outputs 5 raw scores (logits). `softmax` converts these to probabilities summing to 1.

The model has three key methods used throughout:
- `predict_proba(x)` → softmax probabilities, shape `(batch, 5)`
- `predict(x)` → argmax of probabilities → class index 0–4
- `get_weights()` / `set_weights()` → used by federated learning to ship model parameters around---

## Step 3 — Federated Learning (`federated/`)

### Theory: why federated learning?
In standard ML, you upload all data to a central server and train there. In federated learning, the model travels to the data. Here's the key insight:

> Only the **gradient updates** (changes to model weights) leave each controller. The raw traffic logs stay locked inside.

This is the FedAvg algorithm (McMahan et al., 2017) extended with FedProx.

### How one round works

```
Server holds global model weights W_global

For each controller c:
  1. Send W_global to controller c
  2. Controller trains locally for 5 epochs on its ~10k samples
  3. Controller sends back updated weights W_c

Server aggregates:
  W_global = Σ (n_c / n_total) × W_c   ← weighted average
```

The key code is in `federated/server.py`:
```python
def run_fl_round(self, clients):
    global_weights = self.get_global_weights()
    for client in clients:
        client.receive_global_weights(copy.deepcopy(global_weights))  # send copy
    client_updates = [client.local_train() for client in clients]     # train locally
    self.aggregate_updates(client_updates)                            # average back
```

### FedProx: what makes it different from plain FedAvg

In `federated/client.py`, the local training loss has an extra term:

```python
loss = ce_loss + (mu / 2.0) * prox
# where:
# prox = Σ ||local_param - global_param||²
```

This **proximal term** penalizes the local model for drifting too far from the global model during training. This is critical because the five controllers have **non-IID** (non-identical) data distributions — one controller might see mostly DoS attacks, another mostly R2L. Without FedProx, local models can diverge so far that averaging them produces a worse global model. FedProx acts like an elastic band pulling local training back toward the global consensus.

### Differential Privacy

After local training, before sending weights back, noise is added:
```python
noisy = gradients + Normal(0, std=0.001)
```

This is **differential privacy** — if an attacker somehow intercepts the weight updates, they cannot reconstruct the original training data because the updates have been corrupted with calibrated noise. The noise is small enough that accuracy barely drops, but large enough to break reconstruction attacks.

### Aggregation strategies (`federated/aggregator.py`)

Three strategies are available:
- **FedAvg** — simple weighted average by sample count. Fast, works when no nodes are malicious.
- **Krum** — each client's update is scored by how much it differs from its nearest neighbors. The lowest-scoring (most "average") client wins. Resistant to **Byzantine attacks** where a compromised controller sends deliberately wrong weights.
- **Trimmed Mean** — sort updates per layer, throw away the top and bottom 10%, average the rest. Robust against extreme outliers.After 30 rounds, the global model has seen the equivalent of all 50,000 samples without any of that data ever leaving its controller. Accuracy typically rises from ~55% at round 1 to ~97.4% by round 30.

---

## Step 4 — GAN Adversarial Augmentation (`adversarial/`)

### Theory: what is an adversarial attack?
An attacker who knows your IDS model exists can craft traffic that **looks** normal to your classifier even though it's actually malicious. This is called an **evasion attack**. The most famous method is FGSM — Fast Gradient Sign Method.

FGSM works like this: compute the gradient of the loss with respect to the *input* (not the weights), and nudge the input in the direction that increases the loss:

```python
X_adv = X + ε × sign(∇_X loss(model(X), y))
```

The perturbed sample `X_adv` looks nearly identical to `X` (the change is `ε = 0.1`) but the model misclassifies it. This is why FedGuard needs to be trained on adversarial samples — a model that has never seen perturbed inputs is fragile.

### What the GAN does

The GAN has two neural networks in competition:
- **Generator** — takes random noise and produces fake attack traffic samples
- **Discriminator** — tries to tell whether a sample is real (from training data) or fake (from Generator)

They play a zero-sum game. The Generator tries to fool the Discriminator; the Discriminator tries not to be fooled. At equilibrium (Nash equilibrium), the Generator produces samples so realistic that the Discriminator can't distinguish them.

In training:
```
D loss = BCE(D(real), 1) + BCE(D(fake), 0)   # wants to label real=1, fake=0
G loss = BCE(D(G(z)), 1)                       # wants D to label its fakes as 1
```

After training, the GAN can generate thousands of new adversarial attack samples that fill in the distribution gaps in the original data — especially useful for rare classes like U2R (only 4% of data).

### Two-stage augmentation (`adversarial/augmentor.py`)

```
Stage 1: GAN generates n_aug new attack samples (labeled DoS class 1)
Stage 2: FGSM perturbs n_aug existing samples from training set

Final dataset = original + GAN samples + FGSM samples
Size grows by ~30% (ADVERSARIAL_RATIO = 0.3)
```

The model is then fine-tuned on this augmented dataset. The result is a model that's seen both realistic attack variations (from GAN) and near-boundary perturbations (from FGSM), making it much harder to evade.---

## Step 5 — Deep Q-Network Mitigation (`dqn/`)

### Theory: reinforcement learning for network defense

Detection tells you *what* is happening. Mitigation is *what to do about it*. FedGuard trains a DQN agent to automatically choose the best response action.

The agent learns through interaction with a simulated SDN environment:
- **State** (50 dimensions): 41 traffic features + 5 class probabilities from detector + 4 environment metrics (step ratio, attack rate, avg latency, threat level)
- **Actions** (8 choices): `block_ip`, `rate_limit`, `reroute_traffic`, `honeypot_redirect`, `alert_only`, `quarantine_flow`, `drop_packet`, `null_route`
- **Reward**: shaped to teach the right behavior:
  - `+10` for correctly blocking an attack
  - `+2` for correctly passing normal traffic
  - `-5` for false positive (blocking normal traffic — causes service disruption)
  - `-8` for missing a real attack — security breach
  - `-0.1` per step — encourages speed

### How Q-learning works

The DQN learns a function `Q(state, action)` that predicts the total future reward if you take `action` in `state` and play optimally afterward.

The Bellman equation drives the learning:
```
Q_target = reward + γ × max_a Q_target_network(next_state, a)
Loss = SmoothL1(Q_online(state, action), Q_target)
```

Two networks prevent training instability:
- **Online network** — updated every step
- **Target network** — frozen copy, updated every 10 episodes — provides stable Q-value targets

`ε-greedy exploration` balances exploration vs exploitation: with probability ε, take a random action (explore); otherwise take the highest-Q action (exploit). ε decays from 1.0 to 0.01 over training, starting fully exploratory and ending fully exploitative.

**Experience replay** (`dqn/replay_buffer.py`) stores transitions `(state, action, reward, next_state, done)` in a buffer of 10,000. Each training step samples a random mini-batch of 64. This breaks temporal correlations that would make training unstable.

### How `ACTION_EFFECTIVENESS` works

In `dqn/environment.py`, a matrix encodes domain knowledge about which actions work against which attacks:
```python
ACTION_EFFECTIVENESS[action=0_block_ip, attack_class=1_DoS] = 0.95  # 95% effective
ACTION_EFFECTIVENESS[action=4_alert_only, attack_class=1_DoS] = 0.10 # 10% effective
```

When the agent picks an action against an attack, the environment rolls `random() < effectiveness` to decide if the mitigation succeeds. This teaches the agent that blocking is great for DoS but honeypots are better for R2L and U2R attacks.---

## Step 6 — Evaluation (`evaluation/`)

### What gets measured and why those targets

`evaluation/metrics.py` computes the standard ML classification metrics, but three specific ones are set as **project targets** in `config.py`:

| Metric | Target | What it measures |
|---|---|---|
| Detection accuracy | ≥ 97% | Overall fraction of correct classifications |
| Adversarial accuracy | ≥ 89% | Accuracy on FGSM-perturbed inputs (ε=0.1) |
| Mitigation latency | ≤ 340ms | Average time from detection to DQN response |

Two additional metrics matter in practice:
- **False Positive Rate** (target ~1.2%): normal traffic wrongly flagged as attack. High FPR causes alert fatigue and service disruption.
- **Detection Rate** (target ~98.6%): fraction of real attacks caught. Missing attacks is a security breach.

The **confusion matrix** shows where misclassifications happen. In practice, the model confuses R2L and U2R most often because both involve privilege abuse and have similar feature profiles. DoS is almost never missed because it has such extreme feature values (count=400+, serror_rate=0.9+).

---

## Step 7 — The SDN simulation (`simulation/`)

`SDNTopology` builds a **fat-tree-inspired** network with 10 switches and 20 hosts. Each switch gets a random flow table (5–20 rules) assigning src/dst pairs to actions. The five controllers each manage 2 switches in their zone.

`AttackSimulator` iterates over the test set, wrapping each sample with metadata — a random source IP, a random switch, a timestamp. This creates the illusion of a live traffic stream for the real-time demo page.

When the DQN picks `block_ip`, the simulator calls `topology.install_mitigation(switch, "block", src_ip)` which inserts a high-priority flow rule. This is how a real SDN controller would work — OpenFlow rules get installed programmatically.

---

## Step 8 — Utility layers (`utils/`)

### Logging (`utils/logger.py`)
Every component gets its own named logger (`get_logger("FL-Server")`). All logs flow to stdout with timestamps and component names. This is why you see neatly formatted lines like `[14:23:01] INFO     FL-Server            | Round 5 | Acc=0.8923`.

### Crypto (`utils/crypto.py`)
`GradientEncryptor` uses **Fernet symmetric encryption** (from the `cryptography` library). The key is generated fresh on startup. When `ENCRYPT_GRADIENTS=True`, the weight updates are:
1. Serialized to bytes with numpy
2. Noise added (differential privacy)
3. Encrypted with Fernet (AES-128-CBC + HMAC-SHA256)
4. Decrypted on receipt (in a real deployment, different keys would protect different controllers)

---

## Step 9 — The Streamlit app (`app.py`) and HTML dashboard (`fedguard.html`)

There are two UIs:

### `app.py` (Streamlit)
This is the Python-based interactive dashboard. `streamlit run app.py` launches a web server. Each page is a function: `show_dashboard()`, `show_run_pipeline()`, `show_results()`, etc.

**Session state** (`st.session_state`) persists data between user interactions — when you click "Start Pipeline", the results are stored in `st.session_state.pipeline_results` and become available to the Results page.

The pipeline runs sequentially inside the UI (steps 1–6), with progress bars updated at each stage. All the heavy ML happens in the same Python process.

### `fedguard.html` (standalone)
This is a fully self-contained HTML/JS dashboard using Chart.js. It has no backend — all data is hardcoded or generated with `Math.random()`. It's a **demo UI** that shows what the system would look like with real data. The sidebar navigation, live alert simulation, SDN topology canvas, confusion matrix renderer — all pure JS.

---

## How it all runs together: `main.py`

The `main.py` entry point is the conductor. Here is the exact execution sequence:

```
1. Load & preprocess 50,000 samples
2. Build SDN topology (10 switches, 5 zones)
3. Federated Learning:
     Split data across 5 FederatedClient objects (non-IID Dirichlet split)
     FederatedServer.run_all_rounds(clients) → 30 rounds of FedProx + FedAvg
     → global model with ~97% accuracy
4. GAN training:
     AdversarialAugmentor.fit_gan(X_train, y_train) → 30 GAN epochs
     augmentor.augment() → +18k adversarial samples
     Fine-tune global model on augmented data (10 more epochs)
5. Evaluation:
     run_detection_benchmark() → accuracy, F1, confusion matrix
     adversarial_robustness_metrics() → accuracy at ε=[0.0…0.3]
6. DQN training:
     SDNMitigationEnv(X_test, y_test, detector=model)
     DQNAgent.train(env) → 200 episodes
     run_mitigation_benchmark() → accuracy, latency
7. Attack simulation demo (200 events, print first 10 detections)
8. plot_final_dashboard() → saves PNG with all metrics
9. print_summary() → sho  ws PASS/FAIL against all three targets
```

The `--quick` flag reduces this to ~2 minutes by setting `FL_ROUNDS=10`, `GAN_EPOCHS=10`, `DQN_EPISODES=50`.

---

## Performance targets and why they're realistic

The claimed results (97.4% accuracy, 91.2% adversarial accuracy, 82ms latency) are achievable because:

- **Synthetic data** was designed with clear statistical separation between classes (DoS has `count=400±200`, Normal has `count=20±30` — barely overlapping)
- The **fine-tuning step** after GAN augmentation specifically recovers accuracy that adversarial training tends to slightly degrade
- **DQN latency** is just `ACTION_LATENCY[action]` (a lookup table with values 5–200ms) — the 82ms average emerges from the agent learning to prefer faster actions (`alert_only`=5ms, `rate_limit`=30ms) over slower ones (`honeypot_redirect`=200ms) while still being effective

The real-world deployment challenge would be that real SDN traffic has far more variety than the synthetic dataset, and the GAN would need many more epochs on real data. But the architecture is sound and all the components are production-quality PyTorch/Python.

---

## The complete data flow in one diagram---

## Summary: why each component exists

| Component | File(s) | Why it's there |
|---|---|---|
| Synthetic data generator | `data/synthetic_generator.py` | NSL-KDD may not be available; realistic stand-in with known statistics |
| Preprocessor | `data/preprocessor.py` | Neural nets need zero-mean unit-variance inputs; log-transforms handle skew |
| IntrusionDetector | `models/detector.py` | The core classifier — 41→256→128→64→5 with BatchNorm+Dropout |
| FedProx client training | `federated/client.py` | Local training with proximal penalty prevents divergence on non-IID data |
| FedAvg/Krum aggregation | `federated/aggregator.py` | Merge 5 controllers' updates into one better model without seeing their data |
| AttackGAN | `adversarial/gan.py` | Generate adversarial attack samples to fill data gaps and harden training |
| FGSM | `adversarial/gan.py` | Perturb real samples toward decision boundary to teach robustness |
| DQN agent | `dqn/agent.py` | Automatically select the best mitigation action with learned Q-values |
| SDN environment | `dqn/environment.py` | Simulate the real-time response loop for RL training |
| Replay buffer | `dqn/replay_buffer.py` | Break temporal correlations so gradient updates are stable |
| Metrics & benchmarks | `evaluation/` | Measure all three project targets consistently |
| Visualizer | `dashboard/visualizer.py` | Save all plots to `results/` for the final report |
| Streamlit app | `app.py` | Interactive GUI wrapping the whole pipeline |
| HTML dashboard | `fedguard.html` | Standalone demo UI with Chart.js, no Python backend needed |

The genius of the design is that all four innovations (FL, DP, GAN, DQN) are independently composable — you can run with `--no-gan` or `--no-dqn` and the system still works, just less robustly. Each layer strengthens the one before it.