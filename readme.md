# Deep Q-Network (DQN) Implementation for LunarLander-v3

<div align="center">

![LunarLander Training](./lunar_lander_training.png)

**A from-scratch implementation of Deep Q-Network reinforcement learning**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-green.svg)](https://gymnasium.farama.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📋 Overview

This project implements a Deep Q-Network (DQN) agent to solve the **LunarLander-v3** environment from Gymnasium. The implementation is built from scratch using PyTorch, focusing on understanding the mathematical foundations and algorithmic components of value-based reinforcement learning.

**Environment:** The agent learns to control a lunar lander spacecraft, managing thrust and orientation to achieve safe landing between designated flags while minimizing fuel consumption.

---

## 🛠️ Technical Stack

### Core Libraries

- **PyTorch** `2.0+` — Neural network implementation and automatic differentiation
- **Gymnasium** `0.29+` — Reinforcement learning environment interface
- **NumPy** `1.24+` — Numerical computing and array operations
- **Pygame** `2.5+` — Environment rendering and visualization

### Development Tools

- Python `3.8+`
- CUDA (optional, for GPU acceleration)
- Matplotlib (optional, for plotting training curves)

---

## 📁 Project Structure

```
dqn-from-scratch/
│
├── train.py              # Main training loop and hyperparameters
├── agent.py              # DQN agent with policy and learning logic
├── model.py              # Neural network architectures (Q-network)
├── replay_buffer.py      # Experience replay memory implementation
├── evaluate.py           # Agent evaluation and testing script
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── saved_models/         # Directory for trained model checkpoints
```

---

## 🎯 Environment Specifications

**LunarLander-v3** is a classic reinforcement learning benchmark:

| Property | Value |
|----------|-------|
| **State Space** | Continuous (8-dimensional) |
| **Action Space** | Discrete (4 actions) |
| **State Variables** | x, y, vx, vy, angle, angular velocity, leg1 contact, leg2 contact |
| **Actions** | 0: Do nothing, 1: Fire left engine, 2: Fire main engine, 3: Fire right engine |
| **Reward Structure** | Positive for moving toward landing pad, negative for fuel usage and crashes |
| **Episode Termination** | Landing, crash, or leaving boundaries |

**Objective:** Maximize cumulative reward by landing safely between flags with minimal fuel consumption.

---

## 🧮 Mathematical Foundations

### 1. Markov Decision Process (MDP)

The reinforcement learning problem is formalized as an MDP:

$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)
$$

Where:
- $\mathcal{S}$ : State space (8-dimensional continuous vectors)
- $\mathcal{A}$ : Action space (4 discrete actions)
- $\mathcal{P}(s'|s,a)$ : State transition probability distribution
- $\mathcal{R}(s,a,s')$ : Reward function
- $\gamma \in [0,1)$ : Discount factor (temporal preference)

The MDP framework enables optimal sequential decision-making under uncertainty.

---

### 2. Action-Value Function (Q-Function)

The Q-function represents the expected cumulative discounted reward:

$$
Q^\pi(s,a) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty}\gamma^t r_t \mid s_0=s, a_0=a\right]
$$

The optimal Q-function satisfies:

$$
Q^*(s,a) = \max_\pi Q^\pi(s,a)
$$

**Neural Network Approximation:** The Q-function is approximated by a neural network:

$$
Q(s,a;\theta) \approx Q^*(s,a)
$$

where $\theta$ represents the network parameters.

---

### 3. Bellman Optimality Equation

The core recursive relationship in value-based RL:

$$
Q^*(s,a) = \mathbb{E}_{s'}\left[r + \gamma \max_{a'} Q^*(s',a')\right]
$$

This equation states that the optimal value of taking action $a$ in state $s$ equals the immediate reward plus the discounted optimal value of the next state.

**Implementation:** Used to construct training targets for the neural network.

---

### 4. Temporal Difference (TD) Error

The TD error measures the discrepancy between predicted and target Q-values:

$$
\delta_t = r_t + \gamma \max_{a'} Q(s_{t+1},a';\theta^-) - Q(s_t,a_t;\theta)
$$

where $\theta^-$ denotes the target network parameters (explained below).

**Purpose:** Drives gradient-based learning by quantifying prediction error.

---

### 5. Loss Function

The network minimizes the Huber loss (smooth L1 loss):

$$
\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\left[\text{Huber}_\delta(y - Q(s,a;\theta))\right]
$$

where the target is:

$$
y = r + \gamma \max_{a'} Q(s',a';\theta^-)
$$

**Huber Loss Definition:**

$$
\text{Huber}_\delta(x) = \begin{cases}
\frac{1}{2}x^2 & \text{if } |x| \leq \delta \\
\delta(|x| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases}
$$

**Advantages:**
- Quadratic for small errors (smooth gradients)
- Linear for large errors (robust to outliers)
- Stabilizes training compared to MSE

---

### 6. Gradient Descent Optimization

Network parameters are updated using gradient descent:

$$
\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}(\theta)
$$

**Optimizer:** Adam (Adaptive Moment Estimation)
- Combines momentum and RMSprop
- Adaptive learning rates per parameter
- Typical learning rate: $\alpha = 0.001$

**Gradient Clipping:** Gradients are clipped to prevent exploding gradients:

$$
\nabla_\theta \mathcal{L} \leftarrow \text{clip}(\nabla_\theta \mathcal{L}, -1, 1)
$$

---

### 7. Exploration-Exploitation (ε-Greedy Policy)

The agent's behavior policy balances exploration and exploitation:

$$
\pi(a|s) = \begin{cases}
\text{uniform}(\mathcal{A}) & \text{with probability } \varepsilon \\
\arg\max_{a} Q(s,a;\theta) & \text{with probability } 1-\varepsilon
\end{cases}
$$

**Epsilon Decay Schedule:**

$$
\varepsilon_t = \varepsilon_{\text{end}} + (\varepsilon_{\text{start}} - \varepsilon_{\text{end}}) \cdot e^{-\lambda t}
$$

Typical values:
- $\varepsilon_{\text{start}} = 1.0$ (pure exploration)
- $\varepsilon_{\text{end}} = 0.01$ (mostly exploitation)
- $\lambda = 0.995$ (decay rate)

---

### 8. Experience Replay

**Motivation:** Sequential experiences are highly correlated, violating the i.i.d. assumption of stochastic gradient descent.

**Solution:** Store experiences in a replay buffer $\mathcal{D}$ and sample random minibatches.

**Experience Tuple:**

$$
e_t = (s_t, a_t, r_t, s_{t+1}, \text{done}_t)
$$

**Sampling:** Uniformly sample minibatch $\mathcal{B} \sim \mathcal{D}$ of size 64-128.

**Benefits:**
- Breaks temporal correlation
- Enables off-policy learning
- Improves sample efficiency
- Reduces variance in gradient estimates

---

### 9. Target Network Stabilization

**Problem:** Using the same network for prediction and target computation causes instability:

$$
\theta \approx y(\theta) \text{ (self-referential)}
$$

**Solution:** Maintain a separate target network with frozen parameters $\theta^-$:

$$
y = r + \gamma \max_{a'} Q(s',a';\theta^-)
$$

**Update Rule (Hard Update):**

$$
\theta^- \leftarrow \theta \quad \text{every } C \text{ steps}
$$

Typical update frequency: $C = 1000$ steps.

**Purpose:** Stabilizes training by providing fixed targets over multiple updates.

---

## 🔄 Training Algorithm

### DQN Training Loop

```
Initialize replay buffer D with capacity N
Initialize Q-network with random weights θ
Initialize target network θ⁻ = θ

For episode = 1 to M:
    Initialize state s₀
    
    For t = 0 to T:
        # Action Selection
        Select action aₜ using ε-greedy policy
        
        # Environment Interaction
        Execute aₜ, observe reward rₜ and next state sₜ₊₁
        
        # Experience Storage
        Store transition (sₜ, aₜ, rₜ, sₜ₊₁, done) in D
        
        # Learning (if enough samples)
        If |D| > batch_size:
            # Sample Experience
            Sample random minibatch of transitions from D
            
            # Compute Targets
            For each transition:
                yⱼ = rⱼ + γ · max_a' Q(s'ⱼ, a'; θ⁻) · (1 - doneⱼ)
            
            # Gradient Descent
            Perform gradient descent on (yⱼ - Q(sⱼ, aⱼ; θ))²
            
        # Target Network Update
        Every C steps: θ⁻ ← θ
        
        sₜ ← sₜ₊₁
```

---

## 🏗️ Neural Network Architecture

### Q-Network Structure

```
Input Layer (State):        [batch_size, 8]
    ↓
Fully Connected Layer 1:    [8 → 128]
    ↓
ReLU Activation
    ↓
Fully Connected Layer 2:    [128 → 128]
    ↓
ReLU Activation
    ↓
Output Layer (Q-values):    [128 → 4]
```

**Activation Function:** ReLU (Rectified Linear Unit)

$$
\text{ReLU}(x) = \max(0, x)
$$

**Output Interpretation:** The network outputs 4 Q-values, one for each action:

$$
Q(s;\theta) = [Q(s,0), Q(s,1), Q(s,2), Q(s,3)]
$$

---

## ⚙️ Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Learning Rate** | 0.001 | Adam optimizer step size |
| **Discount Factor** (γ) | 0.99 | Future reward weight |
| **Replay Buffer Size** | 100,000 | Maximum stored transitions |
| **Batch Size** | 64 | Minibatch size for training |
| **Target Update Frequency** | 1000 | Steps between target network updates |
| **Initial Epsilon** | 1.0 | Starting exploration rate |
| **Final Epsilon** | 0.01 | Minimum exploration rate |
| **Epsilon Decay** | 0.995 | Exponential decay factor |
| **Hidden Layer Size** | 128 | Neurons per hidden layer |
| **Max Episodes** | 1000 | Training episode limit |

---

## 📈 Learning Dynamics

### Training Phases

**Phase 1: Random Exploration (Episodes 0-100)**
- High epsilon (ε ≈ 1.0)
- Random action selection
- Replay buffer population
- Negative/low rewards
- Q-value initialization

**Phase 2: Early Learning (Episodes 100-400)**
- Decreasing epsilon
- Q-values begin to stabilize
- Agent discovers basic patterns
- Occasional successful landings
- High variance in performance

**Phase 3: Policy Refinement (Episodes 400-800)**
- Low epsilon (ε < 0.1)
- Consistent exploitation
- Smooth landing attempts
- Reduced fuel waste
- Converging Q-values

**Phase 4: Convergence (Episodes 800+)**
- Near-optimal policy
- Stable average rewards (>200)
- Robust to initial conditions
- Minimal exploration

---

## 🚀 Usage

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/dqn-from-scratch.git
cd dqn-from-scratch

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
python train.py
```

**Training Output Example:**
```
Episode 10  | Avg Reward: -151.70 | Epsilon: 0.991 | Loss: 1.4368
Episode 20  | Avg Reward: -170.25 | Epsilon: 0.982 | Loss: 1.0025
Episode 30  | Avg Reward: -200.80 | Epsilon: 0.970 | Loss: 1.3337
...
Episode 500 | Avg Reward: 185.42  | Epsilon: 0.015 | Loss: 0.2104
Episode 600 | Avg Reward: 220.18  | Epsilon: 0.010 | Loss: 0.1567
```

### Evaluation

```bash
python evaluate.py
```

Loads trained model and runs test episodes with visualization.

---

## 📊 Expected Results

After ~600-800 training episodes:

- **Average Reward:** 200-250
- **Success Rate:** >90% safe landings
- **Fuel Efficiency:** Optimized thrust usage
- **Stability:** Smooth descent trajectories

**Performance Criteria:**
- Reward > 200: Excellent landing
- Reward > 100: Successful landing
- Reward < 0: Crash or out-of-bounds

---

## 🔬 Implementation Details

### Key Algorithmic Components

1. **Double Q-Learning Prevention:** Target network prevents overestimation bias
2. **Experience Replay:** Decorrelates sequential samples
3. **Epsilon Decay:** Gradual shift from exploration to exploitation
4. **Huber Loss:** Robust loss function for stable training
5. **Gradient Clipping:** Prevents exploding gradients

### Code Organization

- `agent.py`: Encapsulates DQN logic (action selection, learning, target updates)
- `model.py`: PyTorch neural network definition
- `replay_buffer.py`: Circular buffer implementation with random sampling
- `train.py`: Main training loop with hyperparameters and logging
- `evaluate.py`: Model evaluation with rendering

---

## 🎓 Theoretical Background

### Why Deep Q-Networks?

**Classical Q-Learning Limitations:**
- Requires tabular representation (impractical for continuous/large state spaces)
- Cannot generalize to unseen states

**DQN Advantages:**
- Function approximation via neural networks
- Scales to high-dimensional state spaces
- Generalizes across similar states
- End-to-end learning from raw observations

### Convergence Guarantees

Under certain conditions (tabular case, infinite exploration), Q-learning converges to optimal policy. For neural network approximation:

- **No formal guarantees** due to function approximation and off-policy learning
- **Empirical success** on many benchmarks (Atari, robotics)
- **Stability techniques** (target network, replay buffer) crucial for practical convergence

---

## 🔮 Extensions and Future Work

### Algorithmic Improvements

1. **Double DQN:** Reduces overestimation bias
   $$y = r + \gamma Q(s', \arg\max_{a'} Q(s',a';\theta); \theta^-)$$

2. **Dueling DQN:** Separates state-value and advantage functions
   $$Q(s,a) = V(s) + A(s,a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s,a')$$

3. **Prioritized Experience Replay:** Sample important transitions more frequently

4. **Rainbow DQN:** Combines multiple improvements (Double, Dueling, Prioritized, etc.)

5. **Noisy Networks:** Learned exploration instead of ε-greedy

### Advanced RL Approaches

- **Policy Gradient Methods:** PPO, A3C, TRPO
- **Actor-Critic Methods:** SAC, TD3, DDPG
- **Model-Based RL:** World models, planning
- **Multi-Agent RL:** Competitive/cooperative scenarios

---

## 📚 References

1. **Mnih, V., et al.** (2015). *Human-level control through deep reinforcement learning.* Nature, 518(7540), 529-533.
   - Original DQN paper

2. **Van Hasselt, H., et al.** (2016). *Deep Reinforcement Learning with Double Q-learning.* AAAI.
   - Double DQN improvement

3. **Wang, Z., et al.** (2016). *Dueling Network Architectures for Deep Reinforcement Learning.* ICML.
   - Dueling DQN architecture

4. **Sutton, R. S., & Barto, A. G.** (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
   - Foundational textbook

5. **Gymnasium Documentation:** https://gymnasium.farama.org/
   - Environment specifications

---

## 👤 Author

**[Your Name]**  
BCA Student | Reinforcement Learning & Applied AI

**Focus Areas:**
- Value-based reinforcement learning
- Neural network optimization
- Algorithmic game theory

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` file for details.

---

## 🙏 Acknowledgments

This implementation was developed to deeply understand the mathematical foundations of reinforcement learning, bridging theory and practice through hands-on coding.

Special thanks to:
- DeepMind for pioneering DQN research
- Gymnasium/OpenAI for standardized RL environments
- PyTorch team for excellent deep learning framework

---

<div align="center">

**⭐ If you found this project useful, please consider starring it! ⭐**

Made with 🧠 and ☕

</div>