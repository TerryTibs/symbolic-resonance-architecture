#!/usr/bin/env python3
"""
SRA: ADVANCED MATH EDITION (FIXED)
==================================
Implements the Unified Field Equation:
L = Curiosity - (Reality_Check + Temporal_Prior + Symbolic_Gravity)

Fixes:
- Added .detach() to z_pred and kernel_mem in ResonantEngine
  to prevent 'Backward through graph a second time' errors.
"""

import os
import random
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from sklearn.cluster import KMeans

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
    'grid_size': 100,
    'max_episode_steps': 2000,
    
    # Perception
    'pretrain_samples': 2000,
    'ae_epochs': 50, 
    'ae_lr': 1e-3,
    'latent_dim': 16,
    
    # ADVANCED RESONANCE MATH
    'inference_steps': 5,
    'inference_lr': 0.05,
    'temporal_weight': 0.5,   # Trust the prediction
    'gravity_weight': 0.1,    # Snap to symbols
    
    # Memory
    'memory_threshold': 0.015,
    'memory_capacity': 50000,
    
    # PPO Settings
    'ppo_updates': 1000,        
    'rollout_size': 512,       
    'ppo_lr': 3e-4,
    'gamma': 0.99,
    
    # Tuning
    'curiosity_weight': 0.5,    
    'entropy_coef': 0.01,       
    'living_kernel_strength': 0.2,
    
    'n_symbols_fallback': 8,
    'min_cluster_size': 10 
}

random.seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])
device = torch.device(CONFIG['device'])

if not os.path.exists(CONFIG['output_dir']):
    os.makedirs(CONFIG['output_dir'])

print(f"--- SRA ADVANCED MATH ENGINE INITIALIZED ON {device} ---")

# ==========================
# ENVIRONMENT
# ==========================
class SnakeEnv:
    def __init__(self, size=8):
        self.size = size
        self.reset()

    def reset(self):
        self.snake = [(self.size // 2, self.size // 2)]
        self.place_food()
        self.done = False
        self.steps = 0
        return self._get_obs()

    def place_food(self):
        empty = [(x,y) for x in range(self.size) for y in range(self.size) if (x,y) not in self.snake]
        self.food = random.choice(empty) if empty else None

    def step(self, action):
        if self.done: return self._get_obs(), 0.0, True, {}
        dx, dy = [(-1,0), (0,1), (1,0), (0,-1)][action]
        hx, hy = self.snake[0]
        nx, ny = hx + dx, hy + dy
        self.steps += 1

        if nx < 0 or ny < 0 or nx >= self.size or ny >= self.size or (nx,ny) in self.snake:
            self.done = True
            return self._get_obs(), -2.0, True, {}

        self.snake.insert(0, (nx,ny))
        if self.food and (nx,ny) == self.food:
            self.place_food()
            reward = 10.0 
        else:
            self.snake.pop()
            reward = -0.01

        if self.steps >= CONFIG['max_episode_steps']:
            self.done = True

        return self._get_obs(), reward, self.done, {}

    def _get_obs(self):
        obs = np.zeros((3, self.size, self.size), dtype=np.float32)
        hx, hy = self.snake[0]
        obs[0, hx, hy] = 1.0
        for bx, by in self.snake[1:]:
            obs[1, bx, by] = 1.0
        if self.food:
            fx, fy = self.food
            obs[2, fx, fy] = 1.0
        return obs

# ==========================
# MODULE 1: PERCEPTION
# ==========================
class ConvAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * CONFIG['grid_size']**2, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * CONFIG['grid_size']**2), nn.ReLU(),
            nn.Unflatten(1, (32, CONFIG['grid_size'], CONFIG['grid_size'])),
            nn.ConvTranspose2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, padding=1), nn.Sigmoid()
        )
    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)

# ==========================
# MODULE 2: ADVANCED RESONANCE (FIXED)
# ==========================
class ResonantEngine:
    def __init__(self, model):
        self.model = model
        self.criterion = nn.MSELoss()
    
    def resonate(self, x_img, z_pred=None, kernel_mem=None, steps=None):
        if steps is None: steps = CONFIG['inference_steps']
        
        # --- FIX START ---
        # Detach targets so they act as constants, not variables
        if z_pred is not None: 
            z_pred = z_pred.detach()
        if kernel_mem is not None:
            kernel_mem = kernel_mem.detach()
        # --- FIX END ---
        
        with torch.no_grad():
            z0 = self.model.encoder(x_img)
            
        z = z0.clone().detach().requires_grad_(True)
        opt = optim.Adam([z], lr=CONFIG['inference_lr'])
        final_loss = 0.0
        
        for _ in range(steps):
            opt.zero_grad()
            
            # 1. Reality
            recon = self.model.decoder(z)
            loss_reality = self.criterion(recon, x_img)
            
            # 2. Temporal Prior
            loss_temporal = 0.0
            if z_pred is not None:
                loss_temporal = F.mse_loss(z, z_pred) * CONFIG['temporal_weight']
                
            # 3. Symbolic Gravity
            loss_gravity = 0.0
            if kernel_mem is not None:
                dists = torch.cdist(z, kernel_mem)
                min_dist, _ = torch.min(dists, dim=1)
                loss_gravity = torch.mean(min_dist) * CONFIG['gravity_weight']
            
            total_energy = loss_reality + loss_temporal + loss_gravity
            
            total_energy.backward()
            opt.step()
            final_loss = loss_reality.item()
            
        return z.detach(), final_loss

class LivingKernel:
    def __init__(self, memory, strength=0.3):
        self.memory = memory
        self.strength = strength
        self.mem_tensor = None
    def update(self):
        if len(self.memory) > 0:
            if len(self.memory) > 1000:
                sample = random.sample(self.memory, 1000)
                self.mem_tensor = torch.tensor(np.vstack(sample)).to(device)
            else:
                self.mem_tensor = torch.tensor(np.vstack(self.memory)).to(device)
    def get_bias(self, z):
        if self.mem_tensor is None: return 0.0
        z_norm = F.normalize(z, p=2, dim=1)
        mem_norm = F.normalize(self.mem_tensor, p=2, dim=1)
        similarity = torch.mm(z_norm, mem_norm.t())
        max_sim, _ = torch.max(similarity, dim=1)
        return self.strength * max_sim.unsqueeze(1)

class ForwardPredictor(nn.Module):
    def __init__(self, latent_dim, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 128), nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
    def forward(self, z, action):
        batch_size = z.size(0)
        action_oh = torch.zeros(batch_size, 4).to(z.device)
        action_oh[np.arange(batch_size), action] = 1.0
        inp = torch.cat([z, action_oh], dim=1)
        return self.net(inp)

class ActorCritic(nn.Module):
    def __init__(self, latent_dim, action_dim=4):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU())
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)
    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)

def obs_to_tensor(obs):
    return torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

def save_plot(data, title, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(data)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(CONFIG['output_dir'], filename))
    plt.close()

# ==========================
# MAIN
# ==========================
def main():
    env = SnakeEnv(CONFIG['grid_size'])
    ae = ConvAE(CONFIG['latent_dim']).to(device)
    resonator = ResonantEngine(ae)
    
    print("\n[Phase 1] Pretraining Eyes...")
    data = []
    while len(data) < CONFIG['pretrain_samples']:
        obs = env.reset()
        for _ in range(20):
            obs, _, done, _ = env.step(random.randrange(4))
            data.append(obs)
            if done: break
    data_tensor = torch.tensor(np.stack(data), dtype=torch.float32).to(device)
    opt_ae = optim.Adam(ae.parameters(), lr=CONFIG['ae_lr'])
    ae_losses = []
    
    for ep in range(CONFIG['ae_epochs']):
        perm = torch.randperm(data_tensor.size(0))
        epoch_loss = 0
        for i in range(0, data_tensor.size(0), 64):
            batch = data_tensor[perm[i:i+64]]
            opt_ae.zero_grad()
            _, recon = ae(batch)
            loss = F.mse_loss(recon, batch)
            loss.backward()
            opt_ae.step()
            epoch_loss += loss.item()
        ae_losses.append(epoch_loss / (len(data)/64))
        if (ep+1) % 10 == 0:
            print(f"    Epoch {ep+1}: Loss = {ae_losses[-1]:.5f}")
    save_plot(ae_losses, "Perception Loss", "1_ae_loss.png")

    print("\n[Phase 2] Living Kernel Focus Loop (Advanced Math)...")
    ac = ActorCritic(CONFIG['latent_dim']).to(device)
    fwd = ForwardPredictor(CONFIG['latent_dim']).to(device)
    opt_ppo = optim.Adam(ac.parameters(), lr=CONFIG['ppo_lr'])
    opt_fwd = optim.Adam(fwd.parameters(), lr=1e-3)
    
    memory = []
    kernel = LivingKernel(memory, CONFIG['living_kernel_strength'])
    total_rewards_history = []
    
    z_predicted_next = None 
    
    for update in range(1, CONFIG['ppo_updates'] + 1):
        if update % 10 == 0: kernel.update()
        states, actions, logprobs, rewards, dones, values = [], [], [], [], [], []
        obs = env.reset()
        z_predicted_next = None 
        
        for _ in range(CONFIG['rollout_size']):
            x = obs_to_tensor(obs)
            
            # RESONATE
            z_opt, _ = resonator.resonate(
                x, 
                z_pred=z_predicted_next, 
                kernel_mem=kernel.mem_tensor
            )
            z_curr = z_opt.detach() 
            
            logits, val = ac(z_curr)
            if kernel.mem_tensor is not None:
                logits += kernel.get_bias(z_curr)
            
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            
            next_obs, reward_env, done, _ = env.step(action.item())
            
            x_next = obs_to_tensor(next_obs)
            with torch.no_grad():
                z_next_raw, _ = ae(x_next)
                
            z_predicted_next = fwd(z_curr, action.unsqueeze(0))
            
            intrinsic = F.mse_loss(z_predicted_next, z_next_raw).item()
            total_reward = (intrinsic * CONFIG['curiosity_weight']) + reward_env
            
            states.append(z_curr.detach()) 
            actions.append(action.detach())
            logprobs.append(dist.log_prob(action).detach()) 
            rewards.append(total_reward)
            dones.append(done)
            values.append(val.detach())
            
            loss_fwd = F.mse_loss(z_predicted_next, z_next_raw.detach())
            opt_fwd.zero_grad(); loss_fwd.backward(); opt_fwd.step()
            
            if random.random() < 0.1: 
                with torch.no_grad():
                    rec = ae.decoder(z_curr)
                    l = F.mse_loss(rec, x).item()
                if l < CONFIG['memory_threshold']:
                    memory.append(z_curr.cpu().numpy().flatten())
            
            obs = next_obs
            if done: 
                obs = env.reset()
                z_predicted_next = None
            
        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + CONFIG['gamma'] * (0 if dones[i] else values[i].item()) - values[i].item()
            gae = delta + CONFIG['gamma'] * 0.95 * gae
            returns.insert(0, gae + values[i].item())
        returns = torch.tensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        states_t = torch.cat(states)
        actions_t = torch.stack(actions)
        logprobs_t = torch.stack(logprobs)
        
        for _ in range(4):
            logits, vals = ac(states_t)
            dist = torch.distributions.Categorical(logits=logits)
            new_logprobs = dist.log_prob(actions_t)
            ratio = torch.exp(new_logprobs - logprobs_t)
            adv = returns - vals.squeeze().detach()
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 0.8, 1.2) * adv
            loss = -torch.min(surr1, surr2).mean() + 0.5 * F.mse_loss(vals.squeeze(), returns) - CONFIG['entropy_coef'] * dist.entropy().mean()
            opt_ppo.zero_grad(); loss.backward(); opt_ppo.step()
            
        avg_r = np.mean(rewards)
        total_rewards_history.append(avg_r)
        if update % 20 == 0:
            print(f"  Update {update}/{CONFIG['ppo_updates']}: Avg Reward={avg_r:.4f}")

    save_plot(total_rewards_history, "Training Rewards", "2_training.png")

    print("\n[Saving Model]...")
    memory_numpy = np.vstack(memory) if len(memory) > 0 else np.array([])
    torch.save({
        'ae_state': ae.state_dict(),
        'ac_state': ac.state_dict(),
        'memory': memory_numpy, 
        'config': CONFIG
    }, os.path.join(CONFIG['output_dir'], 'sra_brain.pth'))
    
    print("=== ADVANCED RUN COMPLETE ===")

if __name__ == "__main__":
    main()
