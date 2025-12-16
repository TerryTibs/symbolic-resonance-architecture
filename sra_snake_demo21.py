#!/usr/bin/env python3
"""
SRA: ADVANCED MATH EDITION (FINAL v2)
=====================================
Implements the Lagrangian Objective:
L = Curiosity - (Reality_Check + Temporal_Prior + Symbolic_Gravity)

Fixes:
- Fixed 'float vs Tensor' crash in GAE loop.
- Ensured all PPO calculations stay on device.
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.cluster import KMeans

# Optional HDBSCAN
try:
    import hdbscan
except ImportError:
    hdbscan = None
    print("[System] HDBSCAN not found. Will use KMeans fallback.")

# ==========================
# CONFIGURATION
# ==========================
CONFIG = {
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'output_dir': 'sra_advanced_output',

    # World
    'grid_size': 8,
    'max_episode_steps': 200,

    # Perception
    'pretrain_samples': 2000,
    'ae_epochs': 50,
    'ae_lr': 1e-3,
    'latent_dim': 16,

    # Advanced Resonance
    'inference_steps': 5,
    'inference_lr': 0.05,
    'temporal_weight': 0.5, # Momentum
    'gravity_weight': 0.1,  # Attractor

    # Memory
    'memory_threshold': 0.015,
    'memory_capacity': 50000,
    'living_kernel_strength': 0.2,

    # PPO & Curiosity
    'ppo_updates': 3000,
    'rollout_size': 512,
    'ppo_lr': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'curiosity_weight': 0.5,
    'entropy_coef': 0.01,
    
    # Symbols
    'n_symbols_fallback': 8,
    'min_cluster_size': 10
}

# Setup
random.seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])
device = torch.device(CONFIG['device'])
os.makedirs(CONFIG['output_dir'], exist_ok=True)

print(f"--- SRA ADVANCED MATH ENGINE INITIALIZED ON {device} ---")

# ==========================
# ENVIRONMENT
# ==========================
class SnakeEnv:
    def __init__(self, size):
        self.size = size
        self.reset()

    def reset(self):
        self.snake = [(self.size // 2, self.size // 2)]
        self.place_food()
        self.steps = 0
        self.done = False
        return self._obs()

    def place_food(self):
        cells = [(x,y) for x in range(self.size) for y in range(self.size)
                 if (x,y) not in self.snake]
        self.food = random.choice(cells) if cells else None

    def step(self, action):
        if self.done:
            return self._obs(), 0.0, True, {}

        dx, dy = [(-1,0), (0,1), (1,0), (0,-1)][action]
        hx, hy = self.snake[0]
        nx, ny = hx + dx, hy + dy
        self.steps += 1

        # Crash
        if nx < 0 or ny < 0 or nx >= self.size or ny >= self.size or (nx,ny) in self.snake:
            self.done = True
            return self._obs(), -2.0, True, {}

        self.snake.insert(0, (nx,ny))
        
        # Eat
        if self.food and (nx,ny) == self.food:
            self.place_food()
            r = 10.0
        else:
            self.snake.pop()
            r = -0.01

        if self.steps >= CONFIG['max_episode_steps']:
            self.done = True

        return self._obs(), r, self.done, {}

    def _obs(self):
        obs = np.zeros((3, self.size, self.size), dtype=np.float32)
        hx, hy = self.snake[0]
        obs[0, hx, hy] = 1.0
        for s in self.snake[1:]:
            obs[1, s[0], s[1]] = 1.0
        if self.food:
            fx, fy = self.food
            obs[2, fx, fy] = 1.0
        return obs

# ==========================
# PERCEPTION
# ==========================
class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * CONFIG['grid_size']**2, CONFIG['latent_dim'])
        )
        self.decoder = nn.Sequential(
            nn.Linear(CONFIG['latent_dim'], 32 * CONFIG['grid_size']**2),
            nn.ReLU(),
            nn.Unflatten(1, (32, CONFIG['grid_size'], CONFIG['grid_size'])),
            nn.ConvTranspose2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)

# ==========================
# RESONANCE ENGINE
# ==========================
class ResonantEngine:
    def __init__(self, ae):
        self.ae = ae
        self.mse = nn.MSELoss()

    def resonate(self, x, z_pred=None, mem=None):
        """ The Lagrangian Optimization Loop """
        if z_pred is not None: z_pred = z_pred.detach()
        if mem is not None: mem = mem.detach()

        with torch.no_grad():
            z0 = self.ae.encoder(x)

        z = z0.clone().detach().requires_grad_(True)
        opt = optim.Adam([z], lr=CONFIG['inference_lr'])

        for _ in range(CONFIG['inference_steps']):
            opt.zero_grad()
            recon = self.ae.decoder(z)
            
            # 1. Reality Check
            loss = self.mse(recon, x)

            # 2. Temporal Prior
            if z_pred is not None:
                loss += CONFIG['temporal_weight'] * F.mse_loss(z, z_pred)

            # 3. Symbolic Gravity
            if mem is not None:
                d = torch.cdist(z, mem)
                min_dist, _ = torch.min(d, dim=1)
                loss += CONFIG['gravity_weight'] * torch.mean(min_dist)

            loss.backward()
            opt.step()

        return z.detach()

# ==========================
# LIVING KERNEL
# ==========================
class LivingKernel:
    def __init__(self):
        self.mem = []

    def add(self, z):
        self.mem.append(z)
        if len(self.mem) > CONFIG['memory_capacity']:
            self.mem.pop(0)

    def tensor(self):
        if not self.mem:
            return None
        sample_size = min(1000, len(self.mem))
        sample = random.sample(self.mem, sample_size)
        return torch.tensor(np.vstack(sample), device=device)

# ==========================
# AGENTS
# ==========================
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(CONFIG['latent_dim'], 128), nn.ReLU())
        self.actor = nn.Linear(128, 4)
        self.critic = nn.Linear(128, 1)

    def forward(self, z):
        h = self.shared(z)
        return self.actor(h), self.critic(h)

class ForwardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CONFIG['latent_dim'] + 4, 128),
            nn.ReLU(),
            nn.Linear(128, CONFIG['latent_dim'])
        )

    def forward(self, z, a):
        oh = F.one_hot(a, 4).float()
        return self.net(torch.cat([z, oh], 1))

# ==========================
# UTILS
# ==========================
def save_plot(data, title, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(data)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(CONFIG['output_dir'], filename))
    plt.close()

# ==========================
# TRAINING
# ==========================
def main():
    env = SnakeEnv(CONFIG['grid_size'])
    ae = ConvAE().to(device)
    resonator = ResonantEngine(ae)

    # --- Phase 1: Pretrain AE ---
    print("\n[Phase 1] Pretraining Eyes...")
    data = []
    while len(data) < CONFIG['pretrain_samples']:
        o = env.reset()
        for _ in range(10):
            o, _, d, _ = env.step(random.randrange(4))
            data.append(o)
            if d: break

    data_tensor = torch.tensor(np.stack(data)).to(device)
    opt_ae = optim.Adam(ae.parameters(), lr=CONFIG['ae_lr'])
    ae_losses = []

    for ep in range(CONFIG['ae_epochs']):
        z, recon = ae(data_tensor)
        loss = F.mse_loss(recon, data_tensor)
        opt_ae.zero_grad(); loss.backward(); opt_ae.step()
        ae_losses.append(loss.item())
        if (ep+1) % 10 == 0:
            print(f"  Epoch {ep+1}: Loss = {loss.item():.5f}")
    save_plot(ae_losses, "Perception Loss", "1_ae_loss.png")

    # --- Phase 2: PPO Loop ---
    print("\n[Phase 2] Advanced Resonance PPO Loop...")
    ac = ActorCritic().to(device)
    fm = ForwardModel().to(device)
    opt_ac = optim.Adam(ac.parameters(), lr=CONFIG['ppo_lr'])
    opt_fm = optim.Adam(fm.parameters(), lr=1e-3)

    kernel = LivingKernel()
    reward_history = []

    for update in range(CONFIG['ppo_updates']):
        obs = env.reset()
        mem = kernel.tensor()

        states, acts, logp, rewards, vals = [], [], [], [], []
        z_prev = None

        # Rollout
        for _ in range(CONFIG['rollout_size']):
            x = torch.tensor(obs).unsqueeze(0).to(device)
            
            # Advanced Resonance
            z = resonator.resonate(x, z_prev, mem)
            
            logits, val = ac(z)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()

            obs2, r, d, _ = env.step(a.item())
            
            # Intrinsic Reward
            z2, _ = ae(torch.tensor(obs2).unsqueeze(0).to(device))
            z_pred = fm(z, a)
            intrinsic = F.mse_loss(z_pred, z2).item()
            total_reward = r + CONFIG['curiosity_weight'] * intrinsic

            # Store
            states.append(z)
            acts.append(a)
            logp.append(dist.log_prob(a))
            rewards.append(total_reward)
            vals.append(val.squeeze())

            # Train Forward Model
            opt_fm.zero_grad()
            F.mse_loss(z_pred, z2.detach()).backward()
            opt_fm.step()

            # Memory Gating
            if random.random() < 0.1:
                kernel.add(z.cpu().numpy().flatten())

            obs = obs2 if not d else env.reset()
            z_prev = z_pred.detach()

        # GAE Calculation (FIXED: Using Tensors correctly)
        vals.append(torch.tensor(0.0).to(device))
        returns, adv = [], []
        A = 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + CONFIG['gamma'] * vals[t+1] - vals[t]
            A = delta + CONFIG['gamma'] * CONFIG['gae_lambda'] * A
            adv.insert(0, A)
            # FIX: Returns = Advantage + Value (All Tensors)
            returns.insert(0, A + vals[t])

        adv = torch.stack(adv).detach()
        returns = torch.stack(returns).detach()

        # PPO Update
        logits, v = ac(torch.cat(states))
        dist = torch.distributions.Categorical(logits=logits)
        loss_policy = -(dist.log_prob(torch.cat(acts)) * adv).mean()
        loss_value = 0.5 * F.mse_loss(v.squeeze(), returns)
        loss_entropy = -CONFIG['entropy_coef'] * dist.entropy().mean()
        
        loss = loss_policy + loss_value + loss_entropy

        opt_ac.zero_grad(); loss.backward(); opt_ac.step()

        avg_r = np.mean(rewards)
        reward_history.append(avg_r)
        
        if (update+1) % 20 == 0:
            print(f"  Update {update+1}/{CONFIG['ppo_updates']} | Avg Reward: {avg_r:.3f}")

    save_plot(reward_history, "Training Rewards", "2_training.png")

    # --- Phase 3: Dreams & Saving ---
    print("\n[Phase 3] Dreaming & Saving...")
    
    # Save Model
    torch.save({
        'ae_state': ae.state_dict(),
        'ac_state': ac.state_dict(),
        'memory': np.vstack(kernel.mem) if kernel.mem else [],
        'config': CONFIG
    }, os.path.join(CONFIG['output_dir'], 'sra_brain.pth'))
    
    # Visualize Dreams
    if len(kernel.mem) > 50:
        mem_arr = np.vstack(kernel.mem)
        kmeans = KMeans(n_clusters=CONFIG['n_symbols_fallback'])
        labels = kmeans.fit_predict(mem_arr)
        centers = kmeans.cluster_centers_
        
        fig, axes = plt.subplots(1, min(len(centers), 10), figsize=(15, 2))
        with torch.no_grad():
            for i, ax in enumerate(axes):
                z = torch.tensor(centers[i]).float().unsqueeze(0).to(device)
                recon = ae.decoder(z).squeeze().cpu().numpy()
                img = recon[0] + recon[1]
                ax.imshow(img, cmap='magma')
                ax.axis('off')
        plt.savefig(os.path.join(CONFIG['output_dir'], "3_symbols.png"))

    print("=== ADVANCED SRA COMPLETE ===")

if __name__ == "__main__":
    main()
