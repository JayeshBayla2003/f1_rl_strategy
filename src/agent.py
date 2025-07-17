import math
import random
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.model import DQN # Import the DQN class from model.py

# Define the structure for a single transition (experience) in the replay memory
Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    """
    A cyclic buffer of limited size that stores the transitions observed recently.
    It allows for efficient random sampling of transitions.
    """
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Randomly sample a batch of transitions from memory"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    """
    The main agent that interacts with and learns from the environment.
    """
    def __init__(self, n_observations, n_actions, device):
        # --- Hyperparameters ---
        self.BATCH_SIZE = 128      # Number of transitions sampled from the replay buffer
        self.GAMMA = 0.99          # Discount factor for future rewards
        self.EPS_START = 0.9       # Starting value of epsilon (exploration rate)
        self.EPS_END = 0.05        # Final value of epsilon
        self.EPS_DECAY = 1000      # Controls the rate of exponential decay of epsilon
        self.TAU = 0.005           # The update rate of the target network
        self.LR = 1e-4             # Learning rate of the AdamW optimizer
        
        self.n_actions = n_actions
        self.device = device
        self.steps_done = 0

        # --- Networks ---
        # The policy_net is the main network we train.
        self.policy_net = DQN(n_observations, n_actions).to(self.device)
        # The target_net is a copy of the policy_net. It is updated less frequently,
        # providing a stable target for our loss calculations.
        self.target_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # --- Optimizer and Memory ---
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(10000) # Store up to 10,000 transitions

    def select_action(self, state):
        """
        Selects an action using an epsilon-greedy policy.
        With probability epsilon, a random action is chosen (exploration).
        With probability 1-epsilon, the best action according to the policy network is chosen (exploitation).
        """
        sample = random.random()
        # Epsilon decays over time, moving from exploration to exploitation.
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            # Select a random action from the action space
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        """
        Performs a single step of optimization on the policy network.
        """
        if len(self.memory) < self.BATCH_SIZE:
            return # Don't optimize until we have enough samples in memory

        # Sample a batch of transitions from the replay memory
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for details).
        # This converts a batch-array of Transitions to a Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # --- Q-Value Calculation ---
        # 1. Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        #    columns of actions taken. These are the values we would have chosen
        #    with the policy_net for each state in the batch.
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # 2. Compute V(s_{t+1}) for all next states.
        #    Expected values of actions for non_final_next_states are computed based
        #    on the "older" target_net; selecting their best reward with max(1)[0].
        #    This is merged based on the mask, such that we'll have either the expected
        #    state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        
        # 3. Compute the expected Q values (the "target" for our loss function)
        #    Expected Q Value = reward + (gamma * next_state_value)
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # --- Loss Calculation and Optimization ---
        # Use Huber loss which is less sensitive to outliers than MSELoss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_target_net(self):
        """
        Soft update of the target network's weights:
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)