# %% Import libraries
import numpy as np
from collections import defaultdict
import random

# %% Gridworld Environment
class Gridworld:
    """Implements gridworld dynamics from problem spec"""
    
    def __init__(self):
        self.size = (5, 5)
        self.special = {
            (0,1): {'reward': 10, 'next': (4,1)},  # State A
            (0,3): {'reward': 5, 'next': (2,3)}    # State B
        }
        
        self.actions = ['north', 'south', 'east', 'west']
        self.action_vectors = {
            'north': (-1, 0),
            'south': (1, 0),
            'east': (0, 1),
            'west': (0, -1)
        }
    
    def step(self, state, action):
        """Return (next_state, reward)"""
        # Handle special states
        if state in self.special:
            return self.special[state]['next'], self.special[state]['reward']
        
        # Regular state transition
        di, dj = self.action_vectors[action]
        ni, nj = state[0] + di, state[1] + dj
        
        # Check boundaries
        if 0 <= ni < 5 and 0 <= nj < 5:
            return (ni, nj), 0
        else:
            return state, -1  # Off-grid penalty

# %% Q-Learning Implementation
class QLearner:
    """Implements optimized Q-learning with decay parameters"""
    
    def __init__(self, env):
        self.env = env
        self.q_table = defaultdict(lambda: np.zeros(len(env.actions)))
        self.alpha = 0.2
        self.gamma = 0.9
        self.epsilon = 0.1
        self.action_map = {i:a for i,a in enumerate(env.actions)}
    
    def get_action(self, state, epsilon=None):
        """ε-greedy action selection with decay"""
        epsilon = epsilon or self.epsilon
        if random.random() < epsilon:
            return random.choice(range(len(self.env.actions)))
        return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state):
        """Q-value update with Bellman equation"""
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (
            reward + self.gamma * max_next_q - current_q
        )
    
    def train(self, episodes=5000, steps=5000):
        """Training loop with optional parameter decay"""
        epsilon_decay = self.epsilon / episodes
        for ep in range(episodes):
            state = (np.random.randint(5), np.random.randint(5))
            current_epsilon = self.epsilon - ep * epsilon_decay
            
            for _ in range(steps):
                action_idx = self.get_action(state, current_epsilon)
                next_state, reward = self.env.step(state, self.action_map[action_idx])
                self.update(state, action_idx, reward, next_state)
                state = next_state
    
    def get_policy(self):
        """Extract optimal policy from Q-table"""
        policy = {}
        value = {}
        for i in range(5):
            for j in range(5):
                state = (i,j)
                best_action = np.argmax(self.q_table[state])
                policy[state] = self.env.actions[best_action]
                value[state] = np.max(self.q_table[state])
        return policy, value

# %% Visualization Functions
def print_value_grid(value_dict):
    """Display 5x5 value matrix"""
    print("Optimal Value Function:")
    for i in range(5):
        print(" ".join(f"{value_dict[(i,j)]:.2f}" for j in range(5)))

def print_policy_grid(policy_dict):
    """Display 5x5 policy arrows"""
    arrows = {'north':'↑', 'south':'↓', 'east':'→', 'west':'←'}
    print("\nOptimal Policy:")
    for i in range(5):
        print(" ".join(arrows[policy_dict[(i,j)]] for j in range(5)))

# %% Training and Results
if __name__ == "__main__":
    # Initialize components
    env = Gridworld()
    learner = QLearner(env)
    
    # Train with parameters from assignment
    print("Training Q-learning agent...")
    learner.train(episodes=5000, steps=5000)
    
    # Get and display results
    policy, value = learner.get_policy()
    print_value_grid(value)
    print_policy_grid(policy)