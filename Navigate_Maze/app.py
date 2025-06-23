import numpy as np
import time
from grid_world_env import GridWorldEnv  # Import our custom environment

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.95, 
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration=0.01):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((env.grid_size, env.grid_size, env.action_space.n))
    
    def get_action(self, state):
        # Exploration: random action
        if np.random.random() < self.exploration_rate:
            return self.env.action_space.sample()
        
        # Exploitation: best known action
        return np.argmax(self.q_table[state[0], state[1]])
    
    def update_q_table(self, state, action, reward, next_state):
        # Current Q-value for the state-action pair
        current_q = self.q_table[state[0], state[1], action]
        
        # Maximum Q-value for the next state
        max_next_q = np.max(self.q_table[next_state[0], next_state[1]])
        
        # Q-learning update rule
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        # Update Q-table
        self.q_table[state[0], state[1], action] = new_q
    
    def decay_exploration(self):
        self.exploration_rate = max(
            self.min_exploration, 
            self.exploration_rate * self.exploration_decay
        )

def train_agent(env, agent, episodes=1000, render_every=100, step_delay=0.1):
    rewards_history = []
    steps_history = []
    
    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            # Get action from Q-learning agent
            action = agent.get_action(state)
            
            # Take action in environment
            next_state, reward, done, _ = env.step(action)
            
            # Update Q-table
            agent.update_q_table(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Render occasionally for visualization
            if episode % render_every == 0:
                env.render()
                time.sleep(step_delay)
        
        # Decay exploration rate
        agent.decay_exploration()
        
        # Store metrics
        rewards_history.append(total_reward)
        steps_history.append(steps)
        
        # Print progress
        if episode % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            avg_steps = np.mean(steps_history[-100:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Avg Steps: {avg_steps:.2f}, Exploration: {agent.exploration_rate:.2f}")
    
    return rewards_history, steps_history

def test_agent(env, agent, episodes=5, step_delay=0.5):
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"\nTest Episode {episode}")
        
        while not done:
            # Get action (always exploit, no exploration)
            action = np.argmax(agent.q_table[state[0], state[1]])
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Render and delay for visualization
            env.render()
            time.sleep(step_delay)
            
            print(f"Step {steps}: Position {state}, Action {action}, Reward {reward}")
            
            state = next_state
            total_reward += reward
            steps += 1
        
        print(f"Total Reward: {total_reward}, Total Steps: {steps}")
        time.sleep(1)  # Pause between test episodes

if __name__ == "__main__":
    # Create environment
    env = GridWorldEnv()
    
    # Create Q-learning agent
    agent = QLearningAgent(env)
    
    # Train the agent
    print("Training the agent...")
    train_agent(env, agent, episodes=1000, render_every=100)
    
    # Test the trained agent
    print("\nTesting the trained agent...")
    test_agent(env, agent)
    
    # Close the environment
    env.close()