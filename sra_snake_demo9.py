#!/usr/bin/env python3
"""
SRA CORE FIXED (Robust Version)
-------------------------------
A complete Neuro-Symbolic pipeline implementing:
1. Snake Environment (Sensory Input)
2. ConvAutoencoder (Perception)
3. Resonance Engine (Active Inference/Thinking)
4. Coherence-Gated Memory (Filtering)
5. Symbolic Clustering (Concept Discovery)
6. Graph Export (D3.js Visualization)

Author: SRA Architect
Date: 2025-12-10
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
import networkx as nx
from sklearn.cluster import KMeans

# Safe Import for HDBSCAN (Optional)
try:
    import hdbscan
except ImportError:
    hdbscan = None
    print("[System] HDBSCAN not found. Will strictly use KMeans/Spectral fallback.")

# ------------------------
# CONFIGURATION
# ------------------------
CONFIG = {
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'output_dir': 'sra_core_out',
    'grid_size': 8,
    'pretrain_samples': 1500,
    'ae_epochs': 10,       # Increased slightly for better initial vision
    'ae_lr': 1e-3,
    'latent_dim': 16,
    'inference_steps': 10, # Deeper thinking
    'inference_lr': 0.05,
    'memory_threshold': 0.015,
    'memory_capacity': 2000,
    'min_memory_for_clustering': 50,
    'initial_n_symbols': 4,
}

# ------------------------
# SETUP
# ------------------------
random.seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])
device = torch.device(CONFIG['device'])

OUTDIR = Path(CONFIG['output_dir'])
OUTDIR.mkdir(parents=True, exist_ok=True)

print(f"[System] Initializing SRA on device: {device}")

# ------------------------
# ENVIRONMENT (Snake)
# ------------------------
class SnakeEnv:
    def __init__(self, size=8):
        self.size = size
        self.reset()

    def reset(self):
        self.snake = [(self.size // 2, self.size // 2)]
        self.place_food()
        self.done = False
        self.score = 0
        self.steps = 0
        return self._get_obs()

    def place_food(self):
        empty = [(x,y) for x in range(self.size) for y in range(self.size) if (x,y) not in self.snake]
        self.food = random.choice(empty) if empty else None

    def step(self, action):
        dx,dy = [(-1,0),(0,1),(1,0),(0,-1)][action]
        hx,hy = self.snake[0]
        nx,ny = hx+dx, hy+dy
        self.steps += 1

        if nx < 0 or ny < 0 or nx >= self.size or ny >= self.size or (nx,ny) in self.snake:
            self.done = True
            return self._get_obs(), -1.0, True, {}

        self.snake.insert(0, (nx,ny))
        reward = 0.0
        if self.food is not None and (nx,ny) == self.food:
            reward = 1.0
            self.score += 1
            self.place_food()
        else:
            self.snake.pop()

        if self.steps >= 200: # Shorter episodes for faster gathering
            self.done = True

        return self._get_obs(), reward, False, {}

    def _get_obs(self):
        obs = np.zeros((3, self.size, self.size), dtype=np.float32)
        hx,hy = self.snake[0]
        obs[0, hx, hy] = 1.0 # Head
        for x,y in self.snake[1:]:
            obs[1, x, y] = 1.0 # Body
        if self.food is not None:
            fx,fy = self.food
            obs[2, fx, fy] = 1.0 # Food
        return obs

    def copy_state(self):
        return {'snake': list(self.snake), 'food': self.food, 'done': self.done}

    def set_state(self, st):
        self.snake = list(st['snake'])
        self.food = st['food']
        self.done = st['done']

    def manhattan_head_to_food(self):
        if self.food is None: return None
        hx,hy = self.snake[0]
        fx,fy = self.food
        return abs(hx-fx) + abs(hy-fy)

def obs_to_image_tensor(obs):
    arr = np.array(obs, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 3:
        return torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(device)
    flat = np.array(obs, dtype=np.float32).flatten()
    H = W = CONFIG['grid_size']
    arr = flat.reshape(3, H, W)
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(device)

# ------------------------
# PERCEPTUAL CORE
# ------------------------
class ConvAutoencoder(nn.Module):
    def __init__(self, grid_size=8, latent_dim=16):
        super().__init__()
        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        flat_dim = 32 * grid_size * grid_size
        self.fc_enc = nn.Linear(flat_dim, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, flat_dim)
        self.decoder_conv = nn.Sequential(
            nn.Unflatten(1, (32, grid_size, grid_size)),
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, padding=1), nn.Sigmoid()
        )

    def encoder(self, x):
        h = self.encoder_conv(x)
        return self.fc_enc(h)

    def decoder(self, z):
        h = self.fc_dec(z)
        return self.decoder_conv(h)

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)

# ------------------------
# RESONANT ENGINE
# ------------------------
class ResonantEngine:
    def __init__(self, model):
        self.model = model
        self.criterion = nn.MSELoss()

    def resonate(self, x_img, steps=None, lr=None):
        if steps is None: steps = CONFIG['inference_steps']
        if lr is None: lr = CONFIG['inference_lr']

        # 1. Initial Perception
        with torch.no_grad():
            z0 = self.model.encoder(x_img)
            recon0 = self.model.decoder(z0)
            z0_loss = float(self.criterion(recon0, x_img).item())

        # 2. Meditation Loop (Optimization)
        z = z0.clone().detach().requires_grad_(True)
        opt = optim.Adam([z], lr=lr)
        
        loss_val = z0_loss
        for _ in range(steps):
            opt.zero_grad()
            recon = self.model.decoder(z)
            loss = self.criterion(recon, x_img)
            loss.backward()
            opt.step()
            loss_val = float(loss.item())
            
        return z.detach(), loss_val, z0_loss

# ------------------------
# COGNITIVE MEMORY
# ------------------------
class CognitiveMemory:
    def __init__(self, capacity=2000, threshold=0.012):
        self.capacity = capacity
        self.threshold = threshold
        self.vectors = deque(maxlen=capacity)
        self.coherences = deque(maxlen=capacity)

    def add_event(self, z_tensor, loss, z0_loss=None):
        # Gate: Store if absolute low loss OR significantly improved relative to z0
        store = False
        if loss < self.threshold: store = True
        elif (z0_loss is not None) and (loss < 0.75 * z0_loss): store = True

        if store:
            arr = z_tensor.squeeze(0).cpu().numpy().reshape(-1)
            self.vectors.append(arr)
            self.coherences.append(loss)
            return True
        return False

    def to_array(self):
        if len(self.vectors) == 0: return np.zeros((0, CONFIG['latent_dim']))
        return np.vstack(list(self.vectors))
        
    def __len__(self): return len(self.vectors)

# ------------------------
# SYMBOLIC ABSTRACTION
# ------------------------
def discover_symbols(memory_array, n_symbols=4):
    if memory_array.shape[0] < n_symbols:
        return memory_array, np.zeros(memory_array.shape[0])

    # Try KMeans first
    try:
        kmeans = KMeans(n_clusters=n_symbols, n_init=10, random_state=0)
        labels = kmeans.fit_predict(memory_array)
        centers = kmeans.cluster_centers_
        return centers, labels
    except Exception:
        # Fallback to HDBSCAN if available and KMeans fails (rare)
        if hdbscan:
            print("[Cluster] Falling back to HDBSCAN...")
            clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
            labels = clusterer.fit_predict(memory_array)
            centers = []
            for l in set(labels):
                if l != -1: centers.append(np.mean(memory_array[labels == l], axis=0))
            if not centers: return np.zeros((0,16)), labels
            return np.vstack(centers), labels
        else:
            return np.zeros((0,16)), np.zeros(len(memory_array))

# ------------------------
# GRAPH EXPORT
# ------------------------
class SymbolGraph:
    def __init__(self):
        self.G = nx.Graph()

    def add_symbol(self, idx, coherence=0.0):
        self.G.add_node(f"symbol_{idx}", type="symbol", coherence=float(coherence))

    def add_relation(self, a, b, weight=1.0):
        self.G.add_edge(f"symbol_{a}", f"symbol_{b}", weight=float(weight))

    def export(self, path):
        data = nx.node_link_data(self.G)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[Graph] Exported mind map to {path}")

def flatten_obs_for_ae(obs):
    arr = np.array(obs, dtype=np.float32).flatten()
    return arr

# ------------------------
# MAIN PIPELINE
# ------------------------
def main():
    print("=== SRA CORE STARTING ===")
    env = SnakeEnv(size=CONFIG['grid_size'])
    autoencoder = ConvAutoencoder(grid_size=CONFIG['grid_size'], latent_dim=CONFIG['latent_dim']).to(device)
    
    # 1. Pretraining (Learning to see)
    print("\n[Phase 1] Collecting samples & Pretraining Vision...")
    samples = []
    while len(samples) < CONFIG['pretrain_samples']:
        obs = env.reset()
        for _ in range(30):
            obs, _, done, _ = env.step(random.randrange(4))
            samples.append(flatten_obs_for_ae(obs))
            if done: break
            
    X = np.vstack(samples)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device).view(-1, 3, CONFIG['grid_size'], CONFIG['grid_size'])
    
    opt_ae = optim.Adam(autoencoder.parameters(), lr=CONFIG['ae_lr'])
    for ep in range(1, CONFIG['ae_epochs'] + 1):
        opt_ae.zero_grad()
        _, recon = autoencoder(X_tensor)
        loss = F.mse_loss(recon, X_tensor)
        loss.backward()
        opt_ae.step()
        if ep % 2 == 0: print(f"  > Epoch {ep}: Recon Loss = {loss.item():.5f}")

    # 2. Exploration & Resonance (Learning to think)
    autoencoder.eval()
    resonator = ResonantEngine(autoencoder)
    memory = CognitiveMemory(capacity=CONFIG['memory_capacity'], threshold=CONFIG['memory_threshold'])
    
    print("\n[Phase 2] Exploration & Resonance...")
    for ep in range(1, 26):
        obs = env.reset()
        ep_reward = 0
        
        while True:
            # Plan: Try all actions, resonate on outcome, pick best
            # (Simplified Lookahead)
            best_a = 0
            best_score = -1e9
            state_snap = env.copy_state()
            
            for a in range(4):
                sim = SnakeEnv(env.size)
                sim.set_state(state_snap)
                sim_obs, r, d, _ = sim.step(a)
                
                # The Critical Step: Resonate on the hypothetical future
                x_img = obs_to_image_tensor(sim_obs)
                z_opt, loss_opt, _ = resonator.resonate(x_img, steps=2) # Fast thought
                
                # Score = Reward - Confusion (Loss)
                score = -loss_opt
                if r > 0: score += 5.0
                if d: score -= 10.0
                
                if score > best_score:
                    best_score = score
                    best_a = a
            
            # Act
            obs, reward, done, _ = env.step(best_a)
            ep_reward += reward
            
            # Resonate Deeply on reality & Store
            x_img = obs_to_image_tensor(obs)
            z_opt, loss_opt, z0_loss = resonator.resonate(x_img, steps=CONFIG['inference_steps'])
            memory.add_event(z_opt, loss_opt, z0_loss)
            
            if done: break
            
        if ep % 5 == 0:
            print(f"  > Episode {ep}: Reward={ep_reward:.1f} | Memory Size={len(memory)}")
        if len(memory) >= CONFIG['min_memory_for_clustering']:
            print("  > Sufficient memory gathered.")
            break

    # 3. Symbol Discovery (Dreaming)
    print("\n[Phase 3] Symbol Discovery...")
    mem_arr = memory.to_array()
    if mem_arr.shape[0] > 0:
        centers, labels = discover_symbols(mem_arr, n_symbols=CONFIG['initial_n_symbols'])
        print(f"  > Discovered {centers.shape[0]} unique Symbols (Game States).")
        
        # Save Artifacts
        np.save(OUTDIR / "memory_vectors.npy", mem_arr)
        np.save(OUTDIR / "centers.npy", centers)
        
        # Build Graph
        sg = SymbolGraph()
        for i in range(centers.shape[0]):
            sg.add_symbol(i, coherence=0.9)
            if i > 0: sg.add_relation(i-1, i) # Linear chain for now
        
        sg.export(str(OUTDIR / "symbol_graph.json"))
    else:
        print("  > Not enough memories to dream.")

    print("\n=== SRA PIPELINE COMPLETE ===")
    print(f"Output files located in: {OUTDIR}")

if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Total Runtime: {time.time() - t0:.2f}s")
