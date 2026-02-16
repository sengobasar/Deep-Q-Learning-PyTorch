import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import DQN, DuelingDQN
from replay_buffer import ReplayMemory


# --------------------------------------------------
# DQN Agent
# --------------------------------------------------

class DQNAgent:
    """
    Deep Q-Learning Agent
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        lr=2.5e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=100000,
        buffer_size=100000,
        batch_size=32,
        target_update=1000,
        use_dueling=False
    ):

        # Environment info
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Hyperparameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update

        # Exploration
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Device (CPU / GPU)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Networks
        if use_dueling:
            self.q_net = DuelingDQN(state_dim, action_dim).to(self.device)
            self.target_net = DuelingDQN(state_dim, action_dim).to(self.device)
        else:
            self.q_net = DQN(state_dim, action_dim).to(self.device)
            self.target_net = DQN(state_dim, action_dim).to(self.device)

        # Copy weights
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # Replay memory
        self.memory = ReplayMemory(buffer_size)

        # Step counter
        self.steps = 0


    # --------------------------------------------------
    # Action Selection (Epsilon-Greedy)
    # --------------------------------------------------

    def select_action(self, state, training=True):

        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_net(state)

        return q_values.argmax().item()


    # --------------------------------------------------
    # Store Experience
    # --------------------------------------------------

    def store(self, state, action, reward, next_state, done):

        self.memory.push(state, action, reward, next_state, done)


    # --------------------------------------------------
    # Epsilon Decay
    # --------------------------------------------------

    def update_epsilon(self):

        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start
            - (self.steps / self.epsilon_decay)
            * (self.epsilon_start - self.epsilon_end)
        )


    # --------------------------------------------------
    # Training Step (Bellman Update)
    # --------------------------------------------------

    def train_step(self):

        # Wait until enough data
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch
        transitions = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(
            [t.state for t in transitions]
        ).to(self.device)

        actions = torch.LongTensor(
            [t.action for t in transitions]
        ).unsqueeze(1).to(self.device)

        rewards = torch.FloatTensor(
            [t.reward for t in transitions]
        ).to(self.device)

        next_states = torch.FloatTensor(
            [t.next_state for t in transitions]
        ).to(self.device)

        dones = torch.FloatTensor(
            [t.done for t in transitions]
        ).to(self.device)


        # Q(s,a)
        current_q = self.q_net(states).gather(1, actions)


        # Target: r + γ max Q'(s')
        with torch.no_grad():

            next_q = self.target_net(next_states).max(1)[0]

            target_q = rewards + self.gamma * next_q * (1 - dones)


        # Loss (Huber)
        loss = F.smooth_l1_loss(
            current_q.squeeze(), target_q
        )


        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.q_net.parameters(), 10
        )

        self.optimizer.step()


        # Update target network
        self.steps += 1

        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(
                self.q_net.state_dict()
            )


        # Update epsilon
        self.update_epsilon()


        return loss.item()


    # --------------------------------------------------
    # Save / Load
    # --------------------------------------------------

    def save(self, path):

        torch.save(self.q_net.state_dict(), path)


    def load(self, path):

        self.q_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.q_net.state_dict())
