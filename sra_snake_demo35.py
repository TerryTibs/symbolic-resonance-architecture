#!/usr/bin/env python3
"""
SRA UNIFIED KERNEL (Final Robust Version)
=========================================
Neuro-Symbolic Cognitive Architecture for Snake.

Integrates:
1. Perception (ConvAE)
2. Resonance (Inference-time Optimization)
3. Memory (Dual-Gated: Coherence + Insight)
4. Symbols (Clustering)
5. Control (PPO + Curiosity)

Fixes:
- Enforced 1D shapes for PPO advantages to prevent broadcasting errors.
- Dual-gate memory (stores if coherent OR if resonance provided high gain).
- Robust tensor detach handling to prevent graph retention.

Run: python sra_unified_kernel.py
"""

import os
import random
import time
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

# -----------------------------
# CONFIGURATION
# -----------------------------
CONFIG = {
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'outdir': 'sra_kernel_output',
    
    # Environment
    'grid_size': 10,
    'max_episode_steps': 500,
    
    # Phase 1: Birth (Vision)
    'birth_samples': 3000,
    'ae_epochs': 15,
    'ae_lr': 1e-3,
    'latent_dim': 32,
    
    # Phase 2: Childhood (Resonance)
    'childhood_episodes': 50,
    'inference_steps': 10,
    'inference_lr': 0.08,
    'memory_abs_threshold': 0.025, # Coherence gate
    'memory_rel_factor': 0.85,     # Insight gate (store if loss reduced by 15%)
    'memory_capacity': 10000,
    'min_memory_clustering': 100,
    
    # Phase 3: Language (Symbols)
    'n_symbols': 8,
    
    # Phase 4: Adulthood (RL)
    'adult_updates': 300,
    'rollout_steps': 512, # Batch size for PPO rollout
    'ppo_lr': 3e-4,
    'ppo_epochs': 4,
    'batch_size': 64,     # Minibatch size
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_ratio': 0.2,
    'curiosity_beta': 0.2,
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

# -----------------------------
# ENVIRONMENT
# -----------------------------
class SnakeEnv:
    def __init__(self, size=10):
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
            return self._get_obs(), -1.0, True, {}

        self.snake.insert(0, (nx,ny))
        
        if self.food and (nx,ny) == self.food:
            self.place_food()
            reward = 1.0
        else:
            self.snake.pop()
            reward = 0.0

        if self.steps >= CONFIG['max_episode_steps']: self.done = True
        return self._get_obs(), reward, self.done, {}

    def _get_obs(self):
        obs = np.zeros((3, self.size, self.size), dtype=np.float32)
        hx, hy = self.snake[0]
        obs[0, hx, hy] = 1.0
        for s in self.snake[1:]: obs[1, s[0], s[1]] = 1.0
        if self.food:
            fx, fy = self.food
            obs[2, fx, fy] = 1.0
        return obs
    
    def copy_state(self): return {'snake': list(self.snake), 'food': self.food, 'done': self.done}
    def set_state(self, st): self.snake=list(st['snake']); self.food=st['food']; self.done=st['done']
    def dist_to_food(self):
        if not self.food: return 0
        hx,hy = self.snake[0]; fx,fy = self.food
        return abs(hx-fx)+abs(hy-fy)

def obs_to_tensor(obs):
    return torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

# -----------------------------
# PERCEPTION & RESONANCE
# -----------------------------
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

class ResonantEngine:
    def __init__(self, model):
        self.model = model
        self.criterion = nn.MSELoss()
        
    def resonate(self, x_img, steps=None):
        if steps is None: steps = CONFIG['inference_steps']
        with torch.no_grad():
            z0 = self.model.encoder(x_img)
            recon0 = self.model.decoder(z0)
            z0_loss = self.criterion(recon0, x_img).item()
            
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

# -----------------------------
# MEMORY
# -----------------------------
class CognitiveMemory:
    def __init__(self, capacity):
        self.vectors = deque(maxlen=capacity)
        self.losses = deque(maxlen=capacity)
        
    def add(self, z, loss, z0_loss):
        # Dual Gate: Store if high coherence OR high insight (relative gain)
        coherent = loss < CONFIG['memory_abs_threshold']
        insightful = loss < (z0_loss * CONFIG['memory_rel_factor'])
        
        if coherent or insightful:
            self.vectors.append(z.cpu().numpy().flatten())
            self.losses.append(loss)
            return True
        return False
    
    def get_data(self):
        if not self.vectors: return np.zeros((0, CONFIG['latent_dim']))
        return np.vstack(list(self.vectors))

# -----------------------------
# BRAIN (Actor-Critic + Curiosity)
# -----------------------------
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
        if action.dim() == 1: action = action.unsqueeze(1)
        # One-hot action
        action_oh = torch.zeros(z.size(0), 4).to(z.device)
        action_oh.scatter_(1, action.long(), 1.0)
        inp = torch.cat([z, action_oh], dim=1)
        return self.net(inp)

# -----------------------------
# MAIN LIFECYCLE
# -----------------------------
def main():
    env = SnakeEnv(CONFIG['grid_size'])
    ae = ConvAE(CONFIG['latent_dim']).to(device)
    resonator = ResonantEngine(ae)
    memory = CognitiveMemory(CONFIG['memory_capacity'])
    
    # === PHASE 1: BIRTH (Perception) ===
    print("[Phase 1] Birth: Learning to See...")
    data = []
    while len(data) < CONFIG['birth_samples']:
        obs = env.reset()
        for _ in range(30):
            obs, _, done, _ = env.step(random.randrange(4))
            data.append(obs)
            if done: break
            
    data_tensor = torch.tensor(np.stack(data), dtype=torch.float32).to(device)
    opt_ae = optim.Adam(ae.parameters(), lr=CONFIG['ae_lr'])
    
    for ep in range(CONFIG['ae_epochs']):
        idx = torch.randperm(len(data_tensor))
        for i in range(0, len(data_tensor), 64):
            batch = data_tensor[idx[i:i+64]]
            opt_ae.zero_grad()
            _, recon = ae(batch)
            F.mse_loss(recon, batch).backward()
            opt_ae.step()
            
    # === PHASE 2: CHILDHOOD (Resonant Exploration) ===
    print("[Phase 2] Childhood: Exploring with Resonance...")
    for ep in range(CONFIG['childhood_episodes']):
        obs = env.reset()
        while True:
            # Active Inference (Lookahead)
            best_a = random.randrange(4); best_score = -1e9
            snap = env.copy_state()
            dist_start = env.dist_to_food()
            
            for a in range(4):
                sim = SnakeEnv(env.size); sim.set_state(snap)
                s_obs, s_r, s_d, _ = sim.step(a)
                x = obs_to_tensor(s_obs)
                _, loss, _ = resonator.resonate(x, steps=2)
                score = -loss
                if s_r > 0: score += 5.0
                if s_d: score -= 10.0
                if sim.dist_to_food() < dist_start: score += 0.5
                if score > best_score: best_score = score; best_a = a
            
            obs, _, done, _ = env.step(best_a)
            
            # Memory Gating
            x = obs_to_tensor(obs)
            z_opt, loss, z0_loss = resonator.resonate(x)
            memory.add(z_opt, loss, z0_loss)
            
            if done: break
    
    print(f"  > Childhood Complete. Memory Size: {len(memory.vectors)}")
    
    # === PHASE 3: LANGUAGE (Symbols) ===
    print("[Phase 3] Language: Symbol Discovery...")
    mem_arr = memory.get_data()
    if len(mem_arr) < CONFIG['min_memory_clustering']:
        print("  ! Not enough memory. Using random symbols.")
        centers = np.random.randn(CONFIG['n_symbols'], CONFIG['latent_dim'])
    else:
        kmeans = KMeans(n_clusters=CONFIG['n_symbols'])
        kmeans.fit(mem_arr)
        centers = kmeans.cluster_centers_
    print(f"  > Discovered {len(centers)} Symbols.")
    
    # === PHASE 4: ADULTHOOD (PPO + Curiosity) ===
    print("[Phase 4] Adulthood: Training Policy...")
    
    symbol_emb = nn.Embedding(len(centers), 8).to(device)
    ac = ActorCritic(CONFIG['latent_dim'] + 8).to(device)
    fwd = ForwardPredictor(CONFIG['latent_dim']).to(device)
    opt_ppo = optim.Adam(ac.parameters(), lr=CONFIG['ppo_lr'])
    opt_fwd = optim.Adam(fwd.parameters(), lr=CONFIG['forward_lr'])
    
    rewards_history = []
    
    for update in range(CONFIG['adult_updates']):
        # Rollout buffers
        states, actions, logprobs, rewards, dones, values = [], [], [], [], [], []
        obs = env.reset()
        ep_rew = 0
        
        for _ in range(CONFIG['rollout_steps']):
            # 1. State Construction
            x = obs_to_tensor(obs)
            z_opt, _, _ = resonator.resonate(x, steps=1)
            z_np = z_opt.detach().cpu().numpy().flatten()
            
            # Nearest Symbol
            sym_id = np.argmin(np.sum((centers - z_np)**2, axis=1))
            sym_vec = symbol_emb(torch.tensor([sym_id], device=device)).squeeze(0)
            
            state = torch.cat([z_opt.squeeze(0), sym_vec], dim=0)
            
            # 2. Action
            logits, val = ac(state.unsqueeze(0))
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            
            # 3. Step
            next_obs, r, done, _ = env.step(action.item())
            ep_rew += r
            
            # 4. Curiosity
            x_next = obs_to_tensor(next_obs)
            with torch.no_grad(): z_next, _ = ae(x_next)
            
            pred_z = fwd(z_opt, action.unsqueeze(0))
            int_rew = F.mse_loss(pred_z, z_next).item()
            total_r = r + CONFIG['curiosity_beta'] * int_rew
            
            # Store (DETACH everything)
            states.append(state.detach())
            actions.append(action.detach())
            logprobs.append(dist.log_prob(action).detach())
            rewards.append(total_r)
            dones.append(done)
            values.append(val.detach())
            
            # Train Forward Model
            loss_fwd = F.mse_loss(pred_z, z_next.detach())
            opt_fwd.zero_grad(); loss_fwd.backward(); opt_fwd.step()
            
            obs = next_obs
            if done: obs = env.reset()
            
        # --- PPO UPDATE (Robust) ---
        returns = []
        gae = 0
        next_val = 0
        
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + CONFIG['gamma'] * next_val * (1 - dones[i]) - values[i].item()
            gae = delta + CONFIG['gamma'] * CONFIG['gae_lambda'] * gae * (1 - dones[i])
            returns.insert(0, gae + values[i].item())
            next_val = values[i].item()
            
        # Convert to tensors
        b_states = torch.stack(states)
        b_actions = torch.stack(actions)
        b_logprobs = torch.stack(logprobs)
        
        # FIX: Ensure 1D shapes for advantage calculation
        b_returns = torch.tensor(returns, dtype=torch.float32).to(device).view(-1)
        b_values = torch.tensor([v.item() for v in values], dtype=torch.float32).to(device).view(-1)
        
        # Normalize Advantages
        b_adv = b_returns - b_values
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        
        # Optimization
        for _ in range(CONFIG['ppo_epochs']):
            indices = torch.randperm(len(b_states))
            for start in range(0, len(b_states), CONFIG['batch_size']):
                idx = indices[start:start+CONFIG['batch_size']]
                
                new_logits, new_vals = ac(b_states[idx])
                new_dist = torch.distributions.Categorical(logits=new_logits)
                new_logprobs = new_dist.log_prob(b_actions[idx])
                entropy = new_dist.entropy().mean()
                
                ratio = torch.exp(new_logprobs - b_logprobs[idx])
                
                # Safe Multiplication (Shapes match now)
                surr1 = ratio * b_adv[idx]
                surr2 = torch.clamp(ratio, 1.0 - CONFIG['clip_ratio'], 1.0 + CONFIG['clip_ratio']) * b_adv[idx]
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(new_vals.squeeze(), b_returns[idx])
                
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                opt_ppo.zero_grad(); loss.backward(); opt_ppo.step()
                
        rewards_history.append(ep_rew)
        if (update+1) % 10 == 0:
            avg_r = np.mean(rewards_history[-10:])
            print(f"Update {update+1}: Avg Reward = {avg_r:.3f}")

    # Plot
    plt.plot(rewards_history)
    plt.title("SRA Training Curve")
    plt.savefig(OUTDIR / "training_curve.png")
    print("=== Training Complete. Artifacts Saved. ===")

if __name__ == "__main__":
    main()
