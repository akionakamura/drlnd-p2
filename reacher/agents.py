import numpy as np
import random
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim

from memory import ReplayBuffer
from models import Actor, Critic


DEFAULT_GAMMA = 0.99            # discount factor
DEFAULT_TAU = 1e-3              # for soft update of target parameters
DEFAULT_LR_ACTOR = 1e-4         # learning rate of the actor 
DEFAULT_LR_CRITIC = 3e-4        # learning rate of the critic


class MultiAgent():
    def __init__(
        self,
        n_agents: int,
        state_size: int,
        action_size: int,
        gamma: float = DEFAULT_GAMMA,
        learning_rate_actor: float = DEFAULT_LR_ACTOR,
        learning_rate_critic: float = DEFAULT_LR_CRITIC,
        tau: float = DEFAULT_TAU,
        learn_step: int = 1,
        sync_step: int = 1,
        epsilon_start: float = 1.0,
        min_epsilon: float = 0.01,
        epsilon_decay: float = 0.993
    ):
        self.n_agents = n_agents
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.tau = tau
        self.learn_step = learn_step
        self.sync_step = sync_step
        self.epsilon = self.epsilon_start = epsilon_start
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

        self.actor_local = Actor(self.state_size, self.action_size)
        self.actor_target = Actor(self.state_size, self.action_size)
        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(), lr=self.learning_rate_actor)

        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(), lr=self.learning_rate_critic)

        self.replay_buffer = ReplayBuffer()
        self.t = 0

        self.noise = OUNoise((self.n_agents, action_size), 42)

    def act(self, states, add_noise=True):
        states = torch.from_numpy(np.vstack(states)).float()

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(states).data.numpy()
        self.actor_local.train()

        if add_noise:
            action += self.noise.sample() * self.epsilon

        return np.clip(action, -1, 1)
    
    def step(self, states, actions, rewards, next_states, dones):
        self.replay_buffer.add_multiple(states, actions, rewards, next_states, dones)
        self.t += 1

        # If a multiple of `learn_step`, it is time to learn.
        if self.t % self.learn_step == 0:
            self.learn()

    def learn(self):
        states, actions, rewards, next_states, dones = self.replay_buffer\
            .sample()
        
        # Convert everything to PyTorch tensors.
        states = torch.from_numpy(np.vstack(states)).float()
        actions = torch.from_numpy(np.vstack(actions)).float()
        rewards = torch.from_numpy(np.vstack(rewards)).float().squeeze()        
        next_states = torch.from_numpy(np.vstack(next_states)).float()
        dones = torch.from_numpy(np.vstack(dones).astype(int)).float()\
            .squeeze()

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        if self.t % self.sync_step == 0:
            self.soft_update(self.critic_local, self.critic_target, self.tau)
            self.soft_update(self.actor_local, self.actor_target, self.tau)
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    def episode_finished(self):
        """At the end of an episode, update parameters"""
        # Update epsilon value.
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.uniform(size=x.shape)
        self.state = x + dx
        return self.state