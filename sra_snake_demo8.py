#!/usr/bin/env python3
"""
THE UNIFIED SRA KERNEL (Final Fix)
==================================
A complete Neuro-Symbolic Cognitive Architecture implementation.
Stages 1 through 9 integrated into a single lifecycle.

Fixes:
- Adjusted tensor dimensions in Curiosity Module (Phase 4).
- Relaxed memory thresholds for Phase 2 stability.
- Increased AE epochs for Phase 1 clarity.

Author: Generated for User
Date: 2025-12-09
"""

import os, random, time, math
from pathlib import Path
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'seed': 42,
    'device': 'cpu',  # 'cuda' if available
    'outdir': 'sra_unified_output',
    
    # --- Environment ---
    'env_size': 8,
    'max_steps': 200,

    # --- Phase 1: Birth (Perception) ---
    'birth_samples': 2500,
    'ae_epochs': 100,         # High epochs as requested
    'ae_lr': 1e-3,
    'latent_dim': 16,

    # --- Phase 2: Childhood (Resonance & Memory) ---
    'childhood_episodes': 30, 
    'resonance_steps': 8,
    'resonance_lr': 0.08,
    'memory_abs_threshold': 0.05, # Relaxed threshold
    'memory_capacity': 5000,

    # --- Phase 3: Language (Symbol Grounding) ---
    'initial_symbols': 4, 

    # --- Phase 4: Adulthood (PPO Training) ---
    'adult_updates': 100, 
    'ppo_lr': 3e-4,
    'batch_timesteps': 1024,
    'curiosity_beta': 0.5, 

    # --- Phase 5: Dreaming (Recursive Unity) ---
    'dream_threshold': 0.5, 
}

OUTDIR = Path(CONFIG['outdir'])
OUTDIR.mkdir(parents=True, exist_ok=True)
device = torch.device(CONFIG['device'])

# Seeding
random.seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])

# ==========================================
# STAGE 6: THE BODY (Snake Environment)
# ==========================================
class SnakeEnv:
    def __init__(self, size=8):
        self.size = size
        self.reset()
    def reset(self):
        self.snake = [(self.size//2, self.size//2)]
        self.place_food()
        self.done=False; self.score=0; self.steps=0
        return self._get_obs()
    def place_food(self):
        empty = [(x,y) for x in range(self.size) for y in range(self.size) if (x,y) not in self.snake]
        self.food = random.choice(empty) if empty else None
    def step(self, action):
        dx,dy = [(-1,0),(0,1),(1,0),(0,-1)][action]
        hx,hy = self.snake[0]
        nx,ny = hx+dx, hy+dy
        self.steps += 1
        if nx<0 or ny<0 or nx>=self.size or ny>=self.size or (nx,ny) in self.snake:
            self.done=True
            return self._get_obs(), -10.0, True, {}
        self.snake.insert(0,(nx,ny))
        reward=0.0
        if self.food and (nx,ny)==self.food:
            reward=10.0; self.score+=1; self.place_food()
        else:
            self.snake.pop()
        if self.steps >= CONFIG['max_steps']: self.done = True
        return self._get_obs(), reward, self.done, {}
    def _get_obs(self):
        obs = np.zeros((3,self.size,self.size),dtype=np.float32)
        hx,hy = self.snake[0]; obs[0,hx,hy]=1.0
        for x,y in self.snake[1:]: obs[1,x,y]=1.0
        if self.food: fx,fy = self.food; obs[2,fx,fy]=1.0
        return obs
    def copy_state(self): return {'snake': list(self.snake), 'food': self.food, 'done': self.done}
    def set_state(self, st): self.snake=list(st['snake']); self.food=st['food']; self.done=st['done']
    def dist_to_food(self):
        if not self.food: return 0
        hx,hy = self.snake[0]; fx,fy = self.food; return abs(hx-fx)+abs(hy-fy)

# Utilities
def flatten_obs(obs): return np.array(obs, dtype=np.float32).flatten()
def obs_to_tensor(obs): return torch.tensor(flatten_obs(obs), dtype=torch.float32).unsqueeze(0).to(device)

# ==========================================
# STAGE 1: PERCEPTUAL CORE (Autoencoder)
# ==========================================
class PerceptualCore(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim,256), nn.ReLU(), nn.Linear(256,128), nn.ReLU(), nn.Linear(128, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim,128), nn.ReLU(), nn.Linear(128,256), nn.ReLU(), nn.Linear(256,input_dim), nn.Sigmoid())
    def forward(self,x): z=self.encoder(x); return z, self.decoder(z)

# ==========================================
# STAGE 1 (Part 2): RESONANT ENGINE
# ==========================================
class ResonantEngine:
    def __init__(self, model):
        self.model = model; self.criterion = nn.MSELoss()
    def resonate(self, x, steps=CONFIG['resonance_steps'], lr=CONFIG['resonance_lr']):
        with torch.no_grad():
            z0 = self.model.encoder(x)
        z = z0.clone().detach().requires_grad_(True)
        opt = optim.Adam([z], lr=lr)
        loss_val = 0
        for _ in range(steps):
            opt.zero_grad()
            recon = self.model.decoder(z)
            loss = self.criterion(recon, x)
            loss.backward(); opt.step()
            loss_val = float(loss.item())
        return z.detach(), loss_val

# ==========================================
# STAGE 2: GATED MEMORY
# ==========================================
class CognitiveMemory:
    def __init__(self, capacity):
        self.vectors = deque(maxlen=capacity)
        self.coherences = deque(maxlen=capacity)
    def add_event(self, z, loss):
        if loss < CONFIG['memory_abs_threshold']:
            self.vectors.append(z.detach().cpu().numpy().reshape(-1))
            self.coherences.append(loss)
            return True
        return False
    def to_array(self):
        return np.vstack(list(self.vectors)) if len(self.vectors) > 0 else np.zeros((0, CONFIG['latent_dim']))

# ==========================================
# STAGE 9: PPO AGENT (Brain) & CURIOSITY
# ==========================================
class ActorCritic(nn.Module):
    def __init__(self, input_dim, n_actions=4):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(input_dim,128), nn.ReLU(), nn.Linear(128,128), nn.ReLU())
        self.policy = nn.Sequential(nn.Linear(128, n_actions))
        self.value = nn.Linear(128,1)
    def forward(self, x):
        h = self.shared(x)
        return self.policy(h), self.value(h).squeeze(-1)

class CuriosityModule(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, latent_dim))
    def forward(self, x): return self.net(x)

# ==========================================
# LIFECYCLE MANAGEMENT
# ==========================================

def phase_1_birth(env):
    print("\n--- PHASE 1: BIRTH (Perception) ---")
    print("Agent is blind. Moving randomly to open eyes...")
    data = []
    for _ in range(CONFIG['birth_samples']):
        obs = env.reset()
        for _ in range(30):
            obs, _, done, _ = env.step(random.randrange(4))
            data.append(flatten_obs(obs))
            if done: break
    data = np.vstack(data)
    
    core = PerceptualCore(data.shape[1], CONFIG['latent_dim']).to(device)
    opt = optim.Adam(core.parameters(), lr=CONFIG['ae_lr'])
    crit = nn.MSELoss()
    tensor_data = torch.tensor(data, dtype=torch.float32).to(device)
    
    for ep in range(CONFIG['ae_epochs']):
        opt.zero_grad()
        z, recon = core(tensor_data)
        loss = crit(recon, tensor_data)
        loss.backward()
        opt.step()
        if (ep+1) % 10 == 0:
            print(f"AE Epoch {ep+1}/{CONFIG['ae_epochs']} Loss: {loss.item():.6f}")
    
    return core, data.shape[1]

def phase_2_childhood(env, core, engine, memory):
    print("\n--- PHASE 2: CHILDHOOD (Exploration via Resonance) ---")
    print(f"Agent is exploring using 'Slow Thinking'. Threshold: {CONFIG['memory_abs_threshold']}")
    
    for ep in range(1, CONFIG['childhood_episodes']+1):
        obs = env.reset()
        rewards = 0
        while True:
            # SRA Planning
            best_a = 0; best_score = -1e9
            base_state = env.copy_state()
            dist_before = env.dist_to_food()
            
            for a in range(4):
                sim = SnakeEnv(env.size); sim.set_state(base_state)
                s_obs, s_rew, s_done, _ = sim.step(a)
                x = obs_to_tensor(s_obs)
                _, loss = engine.resonate(x)
                score = -loss 
                if s_rew > 0: score += 10.0
                if s_done: score -= 10.0
                dist_after = sim.dist_to_food()
                if dist_after < dist_before: score += 0.5
                if score > best_score: best_score = score; best_a = a
            
            obs, rew, done, _ = env.step(best_a)
            rewards += rew
            x = obs_to_tensor(obs)
            z_opt, loss_opt = engine.resonate(x)
            memory.add_event(z_opt, loss_opt)
            if done: break
            
        if ep % 5 == 0:
            print(f"Childhood Ep {ep}: Reward={rewards:.1f}, Memories={len(memory.vectors)}")

def phase_3_language(memory):
    print("\n--- PHASE 3: LANGUAGE (Symbol Discovery) ---")
    print("Clustering memories to form Concepts...")
    
    X = memory.to_array()
    
    # SAFETY GUARD
    if len(X) < CONFIG['initial_symbols']:
        print("WARNING: Not enough coherent memories formed.")
        print("Fallback: Creating random seed symbols to continue lifecycle.")
        centers = np.random.randn(CONFIG['initial_symbols'], CONFIG['latent_dim']).astype(np.float32)
    else:
        kmeans = KMeans(n_clusters=CONFIG['initial_symbols'], n_init=10)
        kmeans.fit(X)
        centers = kmeans.cluster_centers_
        
    print(f"Discovered {len(centers)} primal symbols.")
    return centers

def phase_4_adulthood(env, core, engine, centers):
    print("\n--- PHASE 4: ADULTHOOD (PPO + Curiosity) ---")
    print("Agent is learning a Policy using Symbols + Latents...")
    
    k_symbols = len(centers)
    emb_dim = 8
    symbol_emb = nn.Embedding(k_symbols, emb_dim).to(device)
    
    brain = ActorCritic(CONFIG['latent_dim'] + emb_dim, 4).to(device)
    ppo_opt = optim.Adam(brain.parameters(), lr=CONFIG['ppo_lr'])
    fwd_model = CuriosityModule(CONFIG['latent_dim'] + 4, CONFIG['latent_dim']).to(device)
    fwd_opt = optim.Adam(fwd_model.parameters(), lr=1e-3)
    
    def get_state(obs):
        x = obs_to_tensor(obs)
        z, _ = engine.resonate(x)
        z_np = z.detach().cpu().numpy().reshape(-1)
        sym_id = np.argmin(np.sum((centers - z_np)**2, axis=1))
        sym_t = torch.tensor([sym_id], dtype=torch.long).to(device)
        s_emb = symbol_emb(sym_t).squeeze(0)
        return torch.cat([z.squeeze(0), s_emb], dim=0), z, sym_id

    for update in range(1, CONFIG['adult_updates']+1):
        obs = env.reset()
        states, actions, logprobs, rewards, dones = [], [], [], [], []
        
        for _ in range(CONFIG['batch_timesteps']):
            state_t, z_curr, _ = get_state(obs)
            logits, val = brain(state_t.unsqueeze(0))
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            
            obs2, ext_rew, done, _ = env.step(action.item())
            
            state_t2, z_next, _ = get_state(obs2)
            
            # --- FIX APPLIED HERE ---
            # Explicitly squeeze z_curr (1,16 -> 16) to concatenate with act_oh (4)
            # Result is (20), which fwd_model expects
            act_oh = torch.zeros(4).to(device); act_oh[action.item()] = 1.0
            fwd_input = torch.cat([z_curr.squeeze(0), act_oh], dim=0)
            
            pred_z = fwd_model(fwd_input)
            
            # Use squeezed z_next for loss calculation to match pred_z shape
            int_rew = ((pred_z - z_next.squeeze(0))**2).mean().item()
            
            total_rew = ext_rew + (CONFIG['curiosity_beta'] * int_rew)
            
            states.append(state_t); actions.append(action); logprobs.append(dist.log_prob(action))
            rewards.append(total_rew); dones.append(done)
            
            fwd_loss = ((pred_z - z_next.squeeze(0).detach())**2).mean()
            fwd_opt.zero_grad(); fwd_loss.backward(); fwd_opt.step()
            
            obs = obs2
            if done: obs = env.reset()
            
        states = torch.stack(states).detach()
        actions = torch.stack(actions).detach()
        old_logprobs = torch.stack(logprobs).detach()
        
        returns = []; G = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            G = r + 0.99 * G * (1 - d)
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        for _ in range(4): 
            logits, vals = brain(states)
            dist = torch.distributions.Categorical(logits=logits)
            new_logprobs = dist.log_prob(actions)
            ratio = torch.exp(new_logprobs - old_logprobs)
            adv = returns - vals.squeeze()
            
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 0.8, 1.2) * adv
            loss = -torch.min(surr1, surr2).mean() + 0.5 * ((returns - vals)**2).mean()
            
            ppo_opt.zero_grad(); loss.backward(); ppo_opt.step()
            
        if update % 10 == 0:
            avg_rew = sum(rewards) / (sum(dones)+1)
            print(f"Adulthood Update {update}: Avg Reward={avg_rew:.2f}")
            
    return memory

def phase_5_dreaming(memory, centers):
    print("\n--- PHASE 5: DREAMING (Recursive Unity) ---")
    print("Analyzing memories for contradictions...")
    
    X = memory.to_array()
    if len(X) < 10:
        print("Not enough experience to dream.")
        return centers

    labels = np.zeros(len(X))
    distances = np.zeros(len(X))
    
    for i, x in enumerate(X):
        dists = np.sum((centers - x)**2, axis=1)
        labels[i] = np.argmin(dists)
        distances[i] = np.min(dists)
        
    avg_dist = np.mean(distances)
    outliers = X[distances > avg_dist * 1.5]
    
    print(f"Dream Analysis: Found {len(outliers)} contradictory memories.")
    
    if len(outliers) > 20:
        print("Recursive Unity Solver: Synthesizing NEW Symbol...")
        kmeans_dream = KMeans(n_clusters=1)
        kmeans_dream.fit(outliers)
        new_concept = kmeans_dream.cluster_centers_
        
        new_centers = np.vstack([centers, new_concept])
        print(f"Evolution Complete. Symbol Count: {len(centers)} -> {len(new_centers)}")
        return new_centers
    else:
        print("Sleep peaceful. No contradictions found.")
        return centers

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("=== SYMBOLIC RESONANCE ARCHITECTURE: UNIFIED KERNEL ===")
    
    env = SnakeEnv(CONFIG['env_size'])
    core, input_dim = phase_1_birth(env)
    engine = ResonantEngine(core)
    memory = CognitiveMemory(CONFIG['memory_capacity'])
    
    phase_2_childhood(env, core, engine, memory)
    centers = phase_3_language(memory)
    phase_4_adulthood(env, core, engine, centers)
    new_centers = phase_5_dreaming(memory, centers)
    
    print("\nSystem Halted. Saving Brain State...")
    np.save(OUTDIR / "final_symbols.npy", new_centers)
    print(f"Saved {len(new_centers)} symbols to {OUTDIR}")

if __name__ == "__main__":
    main()
