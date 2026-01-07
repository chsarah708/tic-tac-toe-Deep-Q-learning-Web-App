import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from itertools import count


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for Tic-Tac-Toe.
    Takes game state (9 positions) as input and outputs Q-values for 9 possible actions.
    """
    def __init__(self, input_size=9, hidden_size=128, output_size=9):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """Forward pass through the network."""
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayMemory:
    """
    Experience Replay Memory for storing and sampling transitions.
    Stores (state, action, reward, next_state, done) tuples.
    """
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        """Store a transition in memory."""
        self.memory.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        """Sample a random batch of transitions."""
        batch = random.sample(self.memory, min(batch_size, len(self.memory)))
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.memory)


class DQNAgent:
    """
    DQN Agent for playing Tic-Tac-Toe.
    Implements epsilon-greedy exploration and experience replay.
    """
    def __init__(self, state_size=9, action_size=9, learning_rate=0.001, 
                 gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 device=None):
        """
        Initialize the DQN Agent.
        
        Args:
            state_size: Size of the state space (9 for Tic-Tac-Toe)
            action_size: Size of the action space (9 for Tic-Tac-Toe)
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon
            device: Device to run on (CPU or GPU)
        """
        self.device = device if device else torch.device("cpu")
        
        # Network parameters
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        
        # Exploration parameters
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Networks
        self.policy_net = DQNNetwork(input_size=state_size, output_size=action_size).to(self.device)
        self.target_net = DQNNetwork(input_size=state_size, output_size=action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Experience replay memory
        self.memory = ReplayMemory(capacity=10000)
        
    def select_action(self, state, valid_actions=None, training=True):
        """
        Select an action using epsilon-greedy strategy.
        
        Args:
            state: Current game state
            valid_actions: List of valid action indices
            training: Whether in training mode (affects epsilon usage)
            
        Returns:
            Selected action index
        """
        if training and np.random.random() < self.epsilon:
            # Exploration: random valid action
            if valid_actions is not None:
                return random.choice(valid_actions)
            else:
                return random.randint(0, self.action_size - 1)
        else:
            # Exploitation: best action according to policy
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                
                if valid_actions is not None:
                    # Mask invalid actions with very low Q-values
                    q_values_masked = q_values.clone()
                    for i in range(self.action_size):
                        if i not in valid_actions:
                            q_values_masked[0, i] = float('-inf')
                    action = q_values_masked.argmax(dim=1).item()
                else:
                    action = q_values.argmax(dim=1).item()
                    
            return action
    
    def remember(self, state, action, reward, next_state, done):
        """Store transition in replay memory."""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self, batch_size):
        """
        Learn from a batch of experiences.
        
        Args:
            batch_size: Size of the mini-batch to sample
            
        Returns:
            Loss value
        """
        if len(self.memory) < batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # Move tensors to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Compute Q-values for current states
        q_values = self.policy_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.loss_fn(q_values, target_q_values)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with weights from policy network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        """Save model checkpoint."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {filepath}")


def train_agent(env, agent, num_episodes=1000, batch_size=32, target_update_freq=100,
                save_freq=100, save_path='dqn_model.pth'):
    """
    Train the DQN agent on Tic-Tac-Toe environment.
    
    Args:
        env: Game environment with reset() and step(action) methods
        agent: DQNAgent instance
        num_episodes: Number of training episodes
        batch_size: Size of mini-batches for learning
        target_update_freq: Frequency of updating target network
        save_freq: Frequency of saving model checkpoints
        save_path: Path to save model checkpoints
        
    Returns:
        Dictionary with training statistics
    """
    episode_rewards = []
    episode_losses = []
    win_rates = []
    
    print("Starting DQN training...")
    print(f"Total Episodes: {num_episodes}")
    print(f"Batch Size: {batch_size}")
    print(f"Target Update Frequency: {target_update_freq}")
    print("-" * 50)
    
    for episode in range(num_episodes):
        # Reset environment
        state = env.reset()
        total_reward = 0
        total_loss = 0
        loss_count = 0
        done = False
        
        # Episode loop
        while not done:
            # Get valid actions
            valid_actions = env.get_valid_actions()
            
            # Select and perform action
            action = agent.select_action(state, valid_actions=valid_actions, training=True)
            next_state, reward, done, info = env.step(action)
            
            # Store transition in memory
            agent.remember(state, action, reward, next_state, done)
            
            # Learn from experience
            loss = agent.replay(batch_size)
            if loss is not None:
                total_loss += loss
                loss_count += 1
            
            total_reward += reward
            state = next_state
        
        # Update statistics
        episode_rewards.append(total_reward)
        if loss_count > 0:
            avg_loss = total_loss / loss_count
            episode_losses.append(avg_loss)
        
        # Decay exploration rate
        agent.decay_epsilon()
        
        # Update target network
        if (episode + 1) % target_update_freq == 0:
            agent.update_target_network()
        
        # Calculate win rate (last 100 episodes)
        if (episode + 1) % 100 == 0:
            recent_rewards = episode_rewards[-100:]
            win_rate = sum(1 for r in recent_rewards if r > 0) / len(recent_rewards)
            win_rates.append(win_rate)
            
            avg_reward = np.mean(recent_rewards)
            avg_loss = np.mean(episode_losses[-100:]) if len(episode_losses) >= 100 else np.mean(episode_losses)
            
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward (100): {avg_reward:.2f}")
            print(f"  Win Rate (100): {win_rate:.2%}")
            print(f"  Avg Loss: {avg_loss:.4f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print("-" * 50)
        
        # Save model checkpoint
        if (episode + 1) % save_freq == 0:
            agent.save(f"{save_path[:-4]}_ep{episode + 1}.pth")
    
    # Final model save
    agent.save(save_path)
    
    # Print final statistics
    print("\nTraining Complete!")
    print(f"Final Epsilon: {agent.epsilon:.4f}")
    print(f"Average Reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.2f}")
    if win_rates:
        print(f"Final Win Rate: {win_rates[-1]:.2%}")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_losses': episode_losses,
        'win_rates': win_rates
    }


if __name__ == "__main__":
    print("DQN Training Module for Tic-Tac-Toe")
    print("=" * 50)
    print("This module provides:")
    print("  - DQNNetwork: Neural network for Q-value estimation")
    print("  - ReplayMemory: Experience replay buffer")
    print("  - DQNAgent: Agent with epsilon-greedy exploration")
    print("  - train_agent: Main training function")
    print("=" * 50)
