import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import random
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# -----------------------------
# Snake Environment
# -----------------------------
class SnakeEnv:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.snake = deque([[self.grid_size // 2, self.grid_size // 2]])
        self.direction = 1  # 0=up,1=right,2=down,3=left
        self.spawn_food()
        self.done = False
        self.steps = 0
        return self.get_state()

    def spawn_food(self):
        empty = [[x, y] for x in range(self.grid_size) for y in range(self.grid_size) if [x, y] not in self.snake]
        self.food = random.choice(empty)

    def step(self, action):
        if self.done:
            return self.get_state(), 0, True, {}

        # Update direction (prevent 180 turn)
        if (action + 2) % 4 != self.direction:
            self.direction = action

        head = self.snake[0].copy()
        if self.direction == 0: head[1] -= 1
        if self.direction == 1: head[0] += 1
        if self.direction == 2: head[1] += 1
        if self.direction == 3: head[0] -= 1

        # Collision check
        if (head[0] < 0 or head[0] >= self.grid_size or
            head[1] < 0 or head[1] >= self.grid_size or
            head in self.snake):
            self.done = True
            reward = -10
            return self.get_state(), reward, self.done, {}

        self.snake.appendleft(head)

        reward = 0
        if head == self.food:
            reward = 10
            self.spawn_food()
        else:
            self.snake.pop()  # remove tail if no food eaten

        self.steps += 1
        if self.steps >= 200:
            self.done = True

        return self.get_state(), reward, self.done, {}

    def get_state(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for x, y in self.snake:
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                grid[y, x] = 0.5
        fx, fy = self.food
        if 0 <= fx < self.grid_size and 0 <= fy < self.grid_size:
            grid[fy, fx] = 1.0
        return grid.flatten()

# -----------------------------
# Autoencoder
# -----------------------------
class AutoEncoder(nn.Module):
    def __init__(self, input_size, latent_size=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, latent_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

# -----------------------------
# PPO Actor-Critic
# -----------------------------
class ActorCritic(nn.Module):
    def __init__(self, input_size, action_size, latent_size=32):
        super().__init__()
        self.fc = nn.Linear(latent_size, 64)
        self.policy_head = nn.Linear(64, action_size)
        self.value_head = nn.Linear(64, 1)

    def forward(self, z):
        x = F.relu(self.fc(z))
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value

# -----------------------------
# Utilities
# -----------------------------
def collect_random_states(env, count=500):
    states = []
    for _ in range(count):
        obs = env.reset()
        done = False
        while not done:
            action = np.random.randint(0, 4)
            obs, reward, done, _ = env.step(action)
            states.append(obs)
            if len(states) >= count:
                return np.array(states, dtype=np.float32)
    return np.array(states, dtype=np.float32)

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    values = values + [0]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return advantages

# -----------------------------
# Main Training
# -----------------------------
def main():
    grid_size = 10
    env = SnakeEnv(grid_size=grid_size)
    input_size = grid_size * grid_size
    latent_size = 32
    action_size = 4

    # Initialize AE and PPO
    ae = AutoEncoder(input_size, latent_size).to(device)
    ac = ActorCritic(input_size, action_size, latent_size).to(device)

    ae_optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
    ac_optimizer = torch.optim.Adam(ac.parameters(), lr=1e-3)

    # Collect random states for AE pretraining
    print("Collecting random states for AE pretraining...")
    states = collect_random_states(env, 500)
    states_tensor = torch.tensor(states, dtype=torch.float32, device=device)

    # AE Pretraining
    print("Training AE...")
    for epoch in range(8):
        ae_optimizer.zero_grad()
        x_hat, _ = ae(states_tensor)
        loss = F.mse_loss(x_hat, states_tensor)
        loss.backward()
        ae_optimizer.step()
        print(f" AE epoch {epoch+1}/8 loss {loss.item():.6f}")

    # PPO Training loop
    num_iterations = 20
    gamma = 0.99
    eps_clip = 0.2
    for iteration in range(num_iterations):
        obs = env.reset()
        done = False
        states_list, actions_list, rewards_list, values_list, log_probs_list = [], [], [], [], []

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                _, z = ae(obs_tensor)
            logits, value = ac(z)
            dist = Categorical(logits=logits)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action, device=device))

            next_obs, reward, done, _ = env.step(action)

            states_list.append(obs)
            actions_list.append(action)
            rewards_list.append(reward)
            values_list.append(value.item())
            log_probs_list.append(log_prob)

            obs = next_obs

        # Compute advantages and returns
        advantages = compute_gae(rewards_list, values_list, gamma=gamma)
        returns = [a + v for a, v in zip(advantages, values_list)]

        states_tensor = torch.tensor(np.array(states_list, dtype=np.float32), device=device)
        actions_tensor = torch.tensor(actions_list, device=device)
        returns_tensor = torch.tensor(returns, device=device)
        advantages_tensor = torch.tensor(advantages, device=device)

        # Encode states
        _, z_tensor = ae(states_tensor)

        # PPO update
        logits, values = ac(z_tensor)
        dist = Categorical(logits=logits)
        log_probs_new = dist.log_prob(actions_tensor)
        ratio = (log_probs_new - torch.stack(log_probs_list)).exp()
        surr1 = ratio * advantages_tensor
        surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantages_tensor
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values.squeeze(-1), returns_tensor)
        ac_optimizer.zero_grad()
        (policy_loss + value_loss).backward()
        ac_optimizer.step()

        print(f"[Iteration {iteration+1}] PPO update complete")

if __name__ == "__main__":
    t0 = time.time()
    main()
    print("Elapsed:", time.time() - t0)

