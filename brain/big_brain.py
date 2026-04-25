"""
Big Brain — Deep Q-Network Agent (Level 3).

THE final decision-maker.
Receives all sub-agent reports + head agent votes
+ full observation + dissent history + portfolio health
as a 28-dimensional state vector.

Architecture:
  Online network + Target network (Double DQN).
  Experience replay buffer with configurable capacity.
  Epsilon-greedy exploration with decay.
  Gradient clipping for stability.
"""

import os
import random
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class BigBrainNetwork(nn.Module):
    """
    DQN architecture for loan approval.
    Input: 28-dim state vector.
    Output: Q-values for 3 actions (APPROVE, REJECT, REQUEST_MORE_INFO).
    """

    def __init__(self, state_dim: int = 28, action_dim: int = 3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ExperienceReplayBuffer:
    """
    Fixed-size circular buffer for experience replay.
    Stores (state, action, reward, next_state, done) tuples.
    """

    def __init__(self, capacity: int = 10000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int = 64
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch = random.sample(
            list(self.buffer), min(len(self.buffer), batch_size)
        )
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class BigBrainAgent:
    """
    DQN-based reinforcement learning agent.
    Makes the FINAL loan approval decision.

    Uses Double DQN with:
      - Online network for action selection
      - Target network for stable Q-value estimation
      - Epsilon-greedy exploration with decay
      - Experience replay for stable training
    """

    ACTION_MAP = {0: "APPROVE", 1: "REJECT", 2: "REQUEST_MORE_INFO"}
    REVERSE_MAP = {"APPROVE": 0, "REJECT": 1, "REQUEST_MORE_INFO": 2}

    def __init__(
        self,
        state_dim: int = 28,
        action_dim: int = 3,
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        target_update_freq: int = 50,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.online_net = BigBrainNetwork(state_dim, action_dim)
        self.target_net = BigBrainNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(), lr=learning_rate
        )
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ExperienceReplayBuffer()

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.steps = 0
        self.reward_history: List[float] = []
        self.loss_history: List[float] = []

    def decide(self, state_vector: np.ndarray) -> str:
        """
        Select an action using epsilon-greedy policy.
        Returns action string: APPROVE, REJECT, or REQUEST_MORE_INFO.
        """
        if np.random.random() < self.epsilon:
            return self.ACTION_MAP[np.random.randint(3)]

        with torch.no_grad():
            t = torch.FloatTensor(state_vector).unsqueeze(0)
            q = self.online_net(t)
            return self.ACTION_MAP[q.argmax().item()]

    def remember(
        self,
        state: np.ndarray,
        action_str: str,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ) -> None:
        """Store experience in replay buffer."""
        idx = self.REVERSE_MAP[action_str]
        self.replay_buffer.push(state, idx, reward, next_state, done)

    def learn(self) -> Optional[float]:
        """
        Train the online network from a random batch of experiences.
        Updates target network periodically.
        Returns loss value or None if buffer is too small.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = (
            self.replay_buffer.sample(self.batch_size)
        )

        # Current Q-values for chosen actions
        current_q = (
            self.online_net(states)
            .gather(1, actions.unsqueeze(1))
            .squeeze(1)
        )

        # Target Q-values using target network
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online_net.parameters(), 1.0
        )
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(
            self.epsilon_min, self.epsilon * self.epsilon_decay
        )

        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(
                self.online_net.state_dict()
            )

        self.loss_history.append(loss.item())
        return loss.item()

    def save(self, path: str = "brain/big_brain_weights.pt") -> None:
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(
            {
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps": self.steps,
                "reward_history": self.reward_history,
            },
            path,
        )

    def load(self, path: str = "brain/big_brain_weights.pt") -> None:
        """Load model checkpoint if it exists."""
        if not os.path.exists(path):
            return
        ckpt = torch.load(path, weights_only=True)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]
        self.steps = ckpt["steps"]
        self.reward_history = ckpt.get("reward_history", [])
