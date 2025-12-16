# sra_snake_integration.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
CONFIG = {
    'grid_size': 10,       # 10x10 grid
    'latent_dim': 16,      # Size of the "thought"
    'action_space': 4,     # Up, Down, Left, Right
    'resonance_steps': 5,  # How long to "think" before moving
    'gamma': 0.9,          # Discount factor
    'epsilon': 1.0,        # Exploration rate
    'epsilon_decay': 0.995,
    'epsilon_min': 0.01,
    'batch_size': 32,
    'memory_size': 2000
}

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# PART 1: THE BODY (The Snake Environment)
# ==========================================
class SnakeGame:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.snake = deque([(self.grid_size // 2, self.grid_size // 2)])
        self.direction = 1 # Start moving right
        self.food = self.spawn_food()
        self.done = False
        self.steps = 0
        return self.get_state()

    def spawn_food(self):
        while True:
            fx, fy = random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)
            if (fx, fy) not in self.snake:
                return (fx, fy)

    def get_state(self):
        # 0.0 = Empty, 0.5 = Snake, 1.0 = Food
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for x, y in self.snake:
            grid[y, x] = 0.5
        fx, fy = self.food
        grid[fy, fx] = 1.0
        return grid.flatten()

    def step(self, action):
        if self.done:
            return self.get_state(), 0, True, {}

        # Prevent 180 degree turns
        if (action == 0 and self.direction != 2) or \
           (action == 1 and self.direction != 3) or \
           (action == 2 and self.direction != 0) or \
           (action == 3 and self.direction != 1):
            self.direction = action

        # Calculate new head
        head_x, head_y = self.snake[0]
        if self.direction == 0: head_y -= 1 # Up
        elif self.direction == 1: head_x += 1 # Right
        elif self.direction == 2: head_y += 1 # Down
        elif self.direction == 3: head_x -= 1 # Left

        new_head = (head_x, head_y)

        # Collision Check
        if (head_x < 0 or head_x >= self.grid_size or 
            head_y < 0 or head_y >= self.grid_size or 
            new_head in self.snake):
            self.done = True
            return self.get_state(), -10, True, {} # Death penalty

        self.snake.appendleft(new_head)

        reward = 0
        if new_head == self.food:
            reward = 10
            self.food = self.spawn_food()
        else:
            self.snake.pop() # Move tail
            reward = -0.1 # Slight hunger penalty to encourage speed

        self.steps += 1
        if self.steps > 200: # Starvation
            self.done = True
        
        return self.get_state(), reward, self.done, {}

# ==========================================
# PART 2: THE BRAIN (Neuro-Symbolic Core)
# ==========================================
class SRA_Brain(nn.Module):
    def __init__(self, input_dim, latent_dim, action_dim):
        super(SRA_Brain, self).__init__()
        
        # 1. Perception (Encoder)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
        # 2. Imagination (Decoder) - Used for Resonance
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, input_dim), nn.Sigmoid() # Grid values are 0-1
        )
        
        # 3. Decision (Actor/Q-Head) - The new "Hands"
        self.actor = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, action_dim) # Outputs Q-values
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        q_values = self.actor(z)
        return z, recon, q_values

# ==========================================
# PART 3: THE RESONANCE ENGINE
# ==========================================
class ResonantAgent:
    def __init__(self):
        self.input_dim = CONFIG['grid_size'] ** 2
        self.model = SRA_Brain(self.input_dim, CONFIG['latent_dim'], CONFIG['action_space']).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=CONFIG['memory_size'])
        self.mse_loss = nn.MSELoss() # For resonance/reconstruction
        self.huber_loss = nn.SmoothL1Loss() # For RL
    
    def resonate(self, state_tensor):
        """
        Stage 1 Implementation:
        Optimizes the latent vector 'z' against the reality 'x' before acting.
        """
        with torch.no_grad():
            z_initial = self.model.encoder(state_tensor)
        
        # Detach and allow gradients on the thought itself
        z = z_initial.clone().detach().requires_grad_(True)
        opt_z = optim.Adam([z], lr=0.1)
        
        # The Meditation Loop
        for _ in range(CONFIG['resonance_steps']):
            opt_z.zero_grad()
            recon = self.model.decoder(z)
            loss = self.mse_loss(recon, state_tensor)
            loss.backward()
            opt_z.step()
            
        return z.detach()

    def act(self, state):
        # Exploration
        if np.random.rand() <= CONFIG['epsilon']:
            return random.randrange(CONFIG['action_space'])
        
        # Exploitation via Resonance
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # 1. Resonate (Think)
        z_stable = self.resonate(state_tensor)
        
        # 2. Decide (Act) based on the stable thought
        with torch.no_grad():
            q_values = self.model.actor(z_stable)
            
        return torch.argmax(q_values[0]).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < CONFIG['batch_size']:
            return
        
        minibatch = random.sample(self.memory, CONFIG['batch_size'])
        
        states = torch.FloatTensor(np.array([i[0] for i in minibatch])).to(device)
        actions = torch.LongTensor(np.array([i[1] for i in minibatch])).to(device)
        rewards = torch.FloatTensor(np.array([i[2] for i in minibatch])).to(device)
        next_states = torch.FloatTensor(np.array([i[3] for i in minibatch])).to(device)
        dones = torch.FloatTensor(np.array([i[4] for i in minibatch])).to(device)

        # Current Q
        z, recon, current_q = self.model(states)
        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q
        with torch.no_grad():
            _, _, next_q = self.model(next_states)
            max_next_q = next_q.max(1)[0]
            target_q = rewards + (CONFIG['gamma'] * max_next_q * (1 - dones))

        # Combined Loss: RL Accuracy + World Understanding (Reconstruction)
        loss_rl = self.huber_loss(current_q, target_q)
        loss_recon = self.mse_loss(recon, states)
        
        total_loss = loss_rl + loss_recon # The unified objective

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        if CONFIG['epsilon'] > CONFIG['epsilon_min']:
            CONFIG['epsilon'] *= CONFIG['epsilon_decay']
            
        return total_loss.item()

# ==========================================
# MAIN EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    print("--- SRA PHASE 3: ACTION INTEGRATION ---")
    print("Merging Stage 1 (Resonance) with Snake Environment")
    
    env = SnakeGame(grid_size=CONFIG['grid_size'])
    agent = ResonantAgent()
    
    episodes = 200
    scores = []
    
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        loss = 0
        
        while True:
            # 1. Brain: Resonate & Act
            action = agent.act(state)
            
            # 2. Body: Execute & Feel
            next_state, reward, done, _ = env.step(action)
            
            # 3. Memory: Store
            agent.remember(state, action, reward, next_state, done)
            
            # 4. Sleep/Dream: Replay & Train
            loss = agent.replay()
            
            state = next_state
            total_reward += reward
            
            if done:
                scores.append(total_reward)
                print(f"Episode {e+1}/{episodes} | Score: {total_reward:.1f} | Epsilon: {CONFIG['epsilon']:.2f}")
                break
                
    # Visualization of improvement
    plt.plot(scores)
    plt.title("SRA Agent Performance (Resonance + RL)")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.show()
    
    print("Integration Complete. The Agent is thinking and acting.")
