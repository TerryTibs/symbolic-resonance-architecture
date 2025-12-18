#!/usr/bin/env python3
"""
THE UNIFIED SRA KERNEL (Final & Verified)
=========================================
A complete Neuro-Symbolic Cognitive Architecture implementation.
Integrates Perception, Resonance, Memory, Symbols, and Curiosity-driven RL.

Phases:
1. Birth: Pre-training the visual cortex (Autoencoder).
2. Childhood: Exploration via Active Inference (Resonance) to populate memory.
3. Language: Clustering memories to discover Symbols.
4. Adulthood: Training a Policy (PPO) using Symbols + Latents + Curiosity.

Author: SRA Architect
Date: 2025-12-16
"""

import os
import random
import time
import math
import json
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Try importing HDBSCAN, fallback to KMeans if missing
try:
    import hdbscan
except ImportError:
    hdbscan = None
    print("[System] HDBSCAN not found. Using KMeans fallback.")

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'outdir': 'sra_unified_output',
    
    # --- Environment ---
    'grid_size': 8,
    'max_episode_steps': 200,

    # --- Phase 1: Birth (Perception) ---
    'birth_samples': 2500,
    'ae_epochs': 50,
    'ae_lr': 1e-3,
    'latent_dim': 16,

    # --- Phase 2: Childhood (Resonance & Memory) ---
    'childhood_episodes': 30, 
    'inference_steps': 8,
    'inference_lr': 0.08,
    
    # Dual-Gate Memory
    'memory_abs_threshold': 0.02, # Absolute MSE requirement
    'memory_rel_factor': 0.85,    # OR improve by 15% vs z0
    'memory_capacity': 5000,
    'min_memory_clustering': 50,

    # --- Phase 3: Language (Symbol Grounding) ---
    'n_symbols': 8, 

    # --- Phase 4: Adulthood (PPO Training) ---
    'adult_updates': 200, 
    'rollout_steps': 512,
    'ppo_lr': 3e-4,
    'ppo_epochs': 4,
    'batch_size': 64,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_ratio': 0.2,
    
    # Curiosity
    'curiosity_beta': 0.5, # Weight of intrinsic reward
    'forward_lr': 1e-3,
}

# Setup
OUTDIR = Path(CONFIG['outdir'])
OUTDIR.mkdir(parents=True, exist_ok=True)
device = torch.device(CONFIG['device'])

random.seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])

print(f"--- SRA KERNEL INITIALIZED ON {device} ---")

# ==========================================
# ENVIRONMENT
# ==========================================
class SnakeEnv:
    def __init__(self, size=8):
        self.size = size
        self.reset()

    def reset(self):
        self.snake = [(self.size//2, self.size//2)]
        self.place_food()
        self.done = False
        self.steps = 0
        return self._get_obs()

    def place_food(self):
        empty = [(x,y) for x in range(self.size) for y in range(self.size) if (x,y) not in self.snake]
        self.food = random.choice(empty) if empty else None

    def step(self, action):
        if self.done: return self._get_obs(), 0.0, True, {}
        dx, dy = [(-1,0),(0,1),(1,0),(0,-1)][action]
        hx, hy = self.snake[0]
        nx, ny = hx+dx, hy+dy
        self.steps += 1

        # Crash
        if nx<0 or ny<0 or nx>=self.size or ny>=self.size or (nx,ny) in self.snake:
            self.done = True
            return self._get_obs(), -5.0, True, {} # Pain

        self.snake.insert(0, (nx,ny))
        
        # Eat or Move
        if self.food and (nx,ny) == self.food:
            self.place_food()
            reward = 10.0
        else:
            self.snake.pop()
            reward = -0.01 # Step penalty

        if self.steps >= CONFIG['max_episode_steps']: self.done = True
        return self._get_obs(), reward, self.done, {}

    def _get_obs(self):
        obs = np.zeros((3, self.size, self.size), dtype=np.float32)
        hx, hy = self.snake[0]
        obs[0, hx, hy] = 1.0 # Head
        for s in self.snake[1:]: obs[1, s[0], s[1]] = 1.0 # Body
        if self.food:
            fx, fy = self.food
            obs[2, fx, fy] = 1.0 # Food
        return obs
        
    def copy_state(self): return {'snake': list(self.snake), 'food': self.food, 'done': self.done}
    def set_state(self, st): self.snake=list(st['snake']); self.food=st['food']; self.done=st['done']
    def dist_to_food(self):
        if not self.food: return 0
        hx,hy = self.snake[0]; fx,fy = self.food
        return abs(hx-fx)+abs(hy-fy)

def obs_to_tensor(obs):
    return torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

# ==========================================
# MODULE 1: PERCEPTION (Autoencoder)
# ==========================================
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

# ==========================================
# MODULE 2: RESONANCE ENGINE
# ==========================================
class ResonantEngine:
    def __init__(self, model):
        self.model = model
        self.criterion = nn.MSELoss()
        
    def resonate(self, x_img, steps=None):
        if steps is None: steps = CONFIG['inference_steps']
        
        # 1. Fast Perception
        with torch.no_grad():
            z0 = self.model.encoder(x_img)
            recon0 = self.model.decoder(z0)
            z0_loss = self.criterion(recon0, x_img).item()
            
        # 2. Deep Thinking (Optimization)
        z = z0.clone().detach().requires_grad_(True)
        opt = optim.Adam([z], lr=CONFIG['inference_lr'])
        
        final_loss = z0_loss
        for _ in range(steps):
            opt.zero_grad()
            recon = self.model.decoder(z)
            loss = self.criterion(recon, x_img)
            loss.backward()
            opt.step()
            final_loss = loss.item()
            
        return z.detach(), final_loss, z0_loss

# ==========================================
# MODULE 3: COGNITIVE MEMORY
# ==========================================
class CognitiveMemory:
    def __init__(self, capacity):
        self.vectors = deque(maxlen=capacity)
        self.losses = deque(maxlen=capacity)
        
    def add(self, z, loss, z0_loss):
        # Dual Gate: Is it coherent? OR Did thinking help a lot?
        is_coherent = loss < CONFIG['memory_abs_threshold']
        is_insightful = loss < (z0_loss * CONFIG['memory_rel_factor'])
        
        if is_coherent or is_insightful:
            self.vectors.append(z.cpu().numpy().flatten())
            self.losses.append(loss)
            return True
        return False
        
    def get_data(self):
        if len(self.vectors) == 0: return np.zeros((0, CONFIG['latent_dim']))
        return np.vstack(list(self.vectors))

# ==========================================
# MODULE 4: BRAIN (PPO + Curiosity)
# ==========================================
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim=4):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)

class ForwardPredictor(nn.Module):
    def __init__(self, latent_dim, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 64), nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
    def forward(self, z, action):
        # z: [batch, latent], action: [batch]
        if action.dim() == 1: action = action.unsqueeze(1)
        # Create one-hot
        action_oh = torch.zeros(z.size(0), 4).to(z.device)
        action_oh.scatter_(1, action.long(), 1.0)
        
        inp = torch.cat([z, action_oh], dim=1)
        return self.net(inp)

# ==========================================
# UTILS
# ==========================================
def save_plot(data, title, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(data)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(CONFIG['outdir'], filename))
    plt.close()

# ==========================================
# MAIN LIFECYCLE
# ==========================================
def main():
    env = SnakeEnv(CONFIG['grid_size'])
    ae = ConvAE(CONFIG['latent_dim']).to(device)
    resonator = ResonantEngine(ae)
    memory = CognitiveMemory(CONFIG['memory_capacity'])
    
    # --- PHASE 1: BIRTH (Pre-train Eyes) ---
    print("\n[Phase 1] Birth: Learning to See...")
    data = []
    print(f"  > Gathering {CONFIG['birth_samples']} random samples...")
    while len(data) < CONFIG['birth_samples']:
        obs = env.reset()
        for _ in range(20):
            obs, _, done, _ = env.step(random.randrange(4))
            data.append(obs)
            if done: break
            
    data_tensor = torch.tensor(np.stack(data), dtype=torch.float32).to(device)
    opt_ae = optim.Adam(ae.parameters(), lr=CONFIG['ae_lr'])
    
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
        if (ep+1) % 10 == 0:
            print(f"    Epoch {ep+1}: Loss = {epoch_loss/len(data):.5f}")

    # --- PHASE 2: CHILDHOOD (Resonant Exploration) ---
    print("\n[Phase 2] Childhood: Exploring with Resonance...")
    print(f"  > Goal: Fill memory with coherent states.")
    
    for ep in range(1, CONFIG['childhood_episodes'] + 1):
        obs = env.reset()
        rewards = 0
        while True:
            # Active Inference (Simplified Lookahead)
            best_a = random.randrange(4)
            best_score = -float('inf')
            state_snap = env.copy_state()
            dist_start = env.dist_to_food()
            
            # Imagine 4 futures
            for a in range(4):
                sim_env = SnakeEnv(env.size)
                sim_env.set_state(state_snap)
                s_obs, s_rew, s_done, _ = sim_env.step(a)
                
                # Resonate on imagination
                x_sim = obs_to_tensor(s_obs)
                _, loss, _ = resonator.resonate(x_sim, steps=2)
                
                # Score: Low Confusion + Reward + Distance Reduction
                score = -loss
                if s_rew > 0: score += 5.0
                if s_done: score -= 10.0
                if sim_env.dist_to_food() < dist_start: score += 0.5
                
                if score > best_score:
                    best_score = score
                    best_a = a
            
            # Act
            obs, rew, done, _ = env.step(best_a)
            rewards += rew
            
            # Store Memory
            x = obs_to_tensor(obs)
            z_opt, loss_opt, z0_loss = resonator.resonate(x)
            memory.add(z_opt, loss_opt, z0_loss)
            
            if done: break
            
        if ep % 5 == 0:
            print(f"    Ep {ep}: Reward={rewards:.1f} | Memory={len(memory.vectors)}")

    # --- PHASE 3: LANGUAGE (Symbol Discovery) ---
    print("\n[Phase 3] Language: Dreaming Symbols...")
    mem_arr = memory.get_data()
    
    if len(mem_arr) < CONFIG['min_memory_clustering']:
        print("    ! Not enough memories. Filling with random noise for stability.")
        centers = np.random.randn(CONFIG['n_symbols'], CONFIG['latent_dim'])
    else:
        # Use HDBSCAN or KMeans
        if hdbscan:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=1)
            labels = clusterer.fit_predict(mem_arr)
            unique_labels = set(labels) - {-1}
            centers = []
            for l in unique_labels:
                centers.append(np.mean(mem_arr[labels==l], axis=0))
            if not centers: # Fallback if noise
                kmeans = KMeans(n_clusters=CONFIG['n_symbols'])
                kmeans.fit(mem_arr)
                centers = kmeans.cluster_centers_
            else:
                centers = np.vstack(centers)
        else:
            kmeans = KMeans(n_clusters=CONFIG['n_symbols'])
            kmeans.fit(mem_arr)
            centers = kmeans.cluster_centers_
            
    print(f"    > Discovered {len(centers)} Symbols.")
    
    # Save Symbols
    np.save(os.path.join(CONFIG['outdir'], "symbols.npy"), centers)

    # --- PHASE 4: ADULTHOOD (PPO + Curiosity) ---
    print("\n[Phase 4] Adulthood: PPO Training with Curiosity...")
    
    # Embeddings
    symbol_emb_dim = 8
    symbol_embeddings = nn.Embedding(len(centers), symbol_emb_dim).to(device)
    
    # Brains
    input_dim = CONFIG['latent_dim'] + symbol_emb_dim
    ac = ActorCritic(input_dim).to(device)
    fwd = ForwardPredictor(CONFIG['latent_dim']).to(device)
    
    opt_ppo = optim.Adam(ac.parameters(), lr=CONFIG['ppo_lr'])
    opt_fwd = optim.Adam(fwd.parameters(), lr=CONFIG['forward_lr'])
    
    history_rewards = []
    
    for update in range(1, CONFIG['adult_updates'] + 1):
        # Rollout buffers
        states, actions, logprobs, rewards, dones, values = [], [], [], [], [], []
        
        obs = env.reset()
        ep_rew = 0
        
        for _ in range(CONFIG['rollout_steps']):
            # 1. State Construction (Latent + Symbol)
            x = obs_to_tensor(obs)
            z_opt, _, _ = resonator.resonate(x)
            z_np = z_opt.detach().cpu().numpy().flatten()
            
            # Find nearest symbol
            dists = np.sum((centers - z_np)**2, axis=1)
            sym_idx = np.argmin(dists)
            
            sym_tensor = torch.tensor([sym_idx], dtype=torch.long).to(device)
            sym_vec = symbol_embeddings(sym_tensor).squeeze(0)
            
            # Full State
            state = torch.cat([z_opt.squeeze(0), sym_vec], dim=0)
            
            # 2. Action
            logits, val = ac(state.unsqueeze(0))
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            
            # 3. Step
            next_obs, ext_rew, done, _ = env.step(action.item())
            ep_rew += ext_rew
            
            # 4. Curiosity Reward
            x_next = obs_to_tensor(next_obs)
            with torch.no_grad():
                z_next, _ = ae(x_next) # Fast guess
                
            pred_z = fwd(z_opt, action.unsqueeze(0))
            int_rew = F.mse_loss(pred_z, z_next).item()
            
            total_rew = ext_rew + (CONFIG['curiosity_beta'] * int_rew)
            
            # 5. Store (Detach graphs for storage!)
            states.append(state.detach())
            actions.append(action.detach())
            logprobs.append(dist.log_prob(action).detach())
            rewards.append(total_rew)
            dones.append(done)
            values.append(val.detach())
            
            # Train Forward Model Online
            loss_fwd = F.mse_loss(pred_z, z_next.detach())
            opt_fwd.zero_grad(); loss_fwd.backward(); opt_fwd.step()
            
            obs = next_obs
            if done: 
                obs = env.reset()
                
        # --- PPO UPDATE ---
        # Calculate Advantages (GAE)
        returns = []
        gae = 0
        next_val = 0 # bootstrapping simplified
        
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + CONFIG['gamma'] * next_val * (1 - dones[i]) - values[i].item()
            gae = delta + CONFIG['gamma'] * CONFIG['gae_lambda'] * gae * (1 - dones[i])
            returns.insert(0, gae + values[i].item())
            next_val = values[i].item()
            
        returns = torch.tensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Stack batches
        b_states = torch.stack(states)
        b_actions = torch.stack(actions)
        b_logprobs = torch.stack(logprobs)
        b_returns = returns
        b_values = torch.cat(values)
        b_advantages = b_returns - b_values
        
        # Optimize Policy
        for _ in range(CONFIG['ppo_epochs']):
            # Shuffle indices for mini-batches
            indices = torch.randperm(len(b_states))
            for start in range(0, len(b_states), CONFIG['batch_size']):
                end = start + CONFIG['batch_size']
                idx = indices[start:end]
                
                new_logits, new_vals = ac(b_states[idx])
                new_dist = torch.distributions.Categorical(logits=new_logits)
                new_logprobs = new_dist.log_prob(b_actions[idx])
                entropy = new_dist.entropy().mean()
                
                ratio = torch.exp(new_logprobs - b_logprobs[idx])
                surr1 = ratio * b_advantages[idx]
                surr2 = torch.clamp(ratio, 1.0 - CONFIG['clip_ratio'], 1.0 + CONFIG['clip_ratio']) * b_advantages[idx]
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(new_vals.squeeze(), b_returns[idx])
                
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                
                opt_ppo.zero_grad(); loss.backward(); opt_ppo.step()
                
        history_rewards.append(ep_rew)
        if update % 10 == 0:
            avg_r = np.mean(history_rewards[-10:])
            print(f"    Update {update}: Avg Reward = {avg_r:.3f}")

    # Plot
    save_plot(history_rewards, "Training Rewards", "training_curve.png")
    
    # Save Models
    torch.save(ae.state_dict(), os.path.join(OUTDIR, "ae_model.pth"))
    torch.save(ac.state_dict(), os.path.join(OUTDIR, "ppo_model.pth"))
    print("\n=== SYSTEM SAVED. RUN COMPLETE. ===")

if __name__ == "__main__":
    main()
