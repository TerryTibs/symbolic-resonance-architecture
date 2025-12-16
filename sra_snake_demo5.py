import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import time

# ====== Device ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== Hyperparameters ======
state_dim = 200  # Flattened environment state
latent_dim = 32
action_dim = 4   # Snake moves: up, down, left, right
ae_epochs = 8
ppo_epochs = 20
ppo_lr = 1e-3
gamma = 0.99
eps_clip = 0.2
batch_size = 64

# ====== Dummy Snake Environment ======
class DummySnakeEnv:
    def reset(self):
        self.step_count = 0
        return np.random.rand(state_dim)
    
    def step(self, action):
        self.step_count += 1
        reward = random.choice([-10.0, 0.0, 10.0])
        done = self.step_count > 5
        obs = np.random.rand(state_dim)
        return obs, reward, done, {}

env = DummySnakeEnv()

# ====== Autoencoder ======
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

ae = AutoEncoder().to(device)
ae_opt = optim.Adam(ae.parameters(), lr=1e-3)

# ====== PPO Actor-Critic ======
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64)
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)
    
    def forward(self, z):
        x = F.relu(self.fc(z))
        return self.actor(x), self.critic(x)

ac = ActorCritic().to(device)
ac_opt = optim.Adam(ac.parameters(), lr=ppo_lr)

# ====== Helper functions ======
def collect_random_states(env, num_states=500):
    states = []
    while len(states) < num_states:
        obs = env.reset()
        done = False
        while not done:
            action = random.randint(0, action_dim-1)
            obs, _, done, _ = env.step(action)
            states.append(obs)
            if len(states) >= num_states:
                break
    return torch.tensor(np.array(states), dtype=torch.float32, device=device)

def compute_returns(rewards, gamma=0.99):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

def ppo_update(batch_states, batch_actions, batch_returns, batch_old_log_probs):
    # Convert to tensors
    states = torch.tensor(np.array(batch_states), dtype=torch.float32, device=device)
    actions = torch.tensor(batch_actions, dtype=torch.long, device=device)
    returns = torch.tensor(batch_returns, dtype=torch.float32, device=device)
    old_log_probs = torch.tensor(batch_old_log_probs, dtype=torch.float32, device=device)

    # Encode states through AE
    with torch.no_grad():
        z = ae.encoder(states)

    for _ in range(4):  # Multiple PPO epochs
        logits, value = ac(z)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        ratios = torch.exp(log_probs - old_log_probs)
        advantages = returns - value.squeeze().detach()
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(value.squeeze(), returns)
        ac_loss = actor_loss + 0.5 * value_loss
        ac_opt.zero_grad()
        ac_loss.backward()
        ac_opt.step()

# ====== Main Training Loop ======
def main():
    # --- AE Pretraining ---
    print("Collect random states for AE pretraining")
    states = collect_random_states(env, 500)
    print("Train AE")
    for epoch in range(ae_epochs):
        ae_opt.zero_grad()
        recon, z = ae(states)
        loss = F.mse_loss(recon, states)
        loss.backward()
        ae_opt.step()
        print(f" AE epoch {epoch+1}/{ae_epochs} loss {loss.item():.6f}")

    # --- PPO Training ---
    memory = []  # store (state, action, reward, log_prob)
    num_iterations = ppo_epochs
    for it in range(num_iterations):
        obs = env.reset()
        done = False
        states_batch, actions_batch, rewards_batch, log_probs_batch = [], [], [], []

        while not done:
            state_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                z = ae.encoder(state_tensor)
                logits, value = ac(z)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action, device=device))
            obs_next, reward, done, _ = env.step(action)

            states_batch.append(obs)
            actions_batch.append(action)
            rewards_batch.append(reward)
            log_probs_batch.append(log_prob.item())

            obs = obs_next

        returns = compute_returns(rewards_batch, gamma)
        ppo_update(states_batch, actions_batch, returns, log_probs_batch)
        print(f"[Iteration {it+1}] PPO update complete")

if __name__ == "__main__":
    t0 = time.time()
    main()
    print("Elapsed:", time.time()-t0)

