#!/usr/bin/env python3
"""
SRA CORE — Improved gating & diagnostics

Fixes applied:
- Accept memories when either loss < abs_threshold OR loss < rel_factor * z0_loss
- Raised abs_threshold (less strict) and added rel_factor
- Increased resonance steps (more chance to find coherent z)
- Increased exploration episodes and added debug prints when memory stored
- Kept ConvAE input-shape fixes from before

Run: python sra_core_fixed_improved.py
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
import hdbscan
from sklearn.cluster import KMeans

# ------------------------
# CONFIGURATION (tuned)
# ------------------------
CONFIG = {
    'seed': 42,
    'device': 'cpu',                # 'cuda' if available
    'output_dir': 'sra_core_out_improved',
    'grid_size': 8,                 # grid size
    'pretrain_samples': 1200,       # fewer samples for quick runs
    'ae_epochs': 100,                 # fewer epochs to iterate faster
    'ae_lr': 1e-3,
    'latent_dim': 16,

    # resonance / inference
    'inference_steps': 12,          # more opt steps to stabilize z
    'inference_lr': 0.08,

    # memory gating: ABS OR REL acceptance
    'memory_abs_threshold': 0.05,   # less strict absolute threshold
    'memory_rel_factor': 0.90,      # accept if z* < rel_factor * z0_loss
    'memory_capacity': 5000,

    # exploration & clustering
    'exploration_episodes': 10000,     # increase exploration
    'min_memory_for_clustering': 30,
    'initial_n_symbols': 4,
}

# ------------------------
# SETUP
# ------------------------
random.seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])
device = torch.device(CONFIG['device'])

OUTDIR = Path(CONFIG['output_dir']); OUTDIR.mkdir(parents=True, exist_ok=True)

# ------------------------
# ENVIRONMENT
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
            return self._get_obs(), -10.0, True, {}

        self.snake.insert(0,(nx,ny))
        reward = 0.0
        if self.food is not None and (nx,ny) == self.food:
            reward = 10.0
            self.score += 1
            self.place_food()
        else:
            self.snake.pop()

        if self.steps >= 1000:
            self.done = True

        return self._get_obs(), reward, False, {}

    def _get_obs(self):
        obs = np.zeros((3,self.size,self.size), dtype=np.float32)
        hx,hy = self.snake[0]
        obs[0,hx,hy] = 1.0
        for x,y in self.snake[1:]:
            obs[1,x,y] = 1.0
        if self.food is not None:
            fx,fy = self.food
            obs[2,fx,fy] = 1.0
        return obs

    def copy_state(self):
        return {'snake': list(self.snake), 'food': self.food, 'done': self.done}

    def set_state(self, st):
        self.snake = list(st['snake'])
        self.food = st['food']
        self.done = st['done']

    def manhattan_head_to_food(self):
        if self.food is None:
            return None
        hx,hy = self.snake[0]
        fx,fy = self.food
        return abs(hx-fx) + abs(hy-fy)

# ------------------------
# UTIL: convert obs -> (1,3,H,W) tensor
# ------------------------
def obs_to_image_tensor(obs):
    arr = np.array(obs, dtype=np.float32)
    H = W = CONFIG['grid_size']
    if arr.shape == (3, H, W):
        return torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(device)
    flat = arr.flatten()
    if flat.size != 3*H*W:
        raise ValueError(f"Unexpected obs size {flat.size}, expected {3*H*W}")
    arr = flat.reshape(3, H, W)
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(device)

# ------------------------
# Conv Autoencoder (same as before)
# ------------------------
class ConvAutoencoder(nn.Module):
    def __init__(self, grid_size=8, latent_dim=16):
        super().__init__()
        C = 3; H = grid_size; W = grid_size
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(C, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        flat_dim = 32 * H * W
        self.fc_enc = nn.Linear(flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat_dim)
        self.decoder_conv = nn.Sequential(
            nn.Unflatten(1, (32, H, W)),
            nn.ConvTranspose2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, C, 3, padding=1), nn.Sigmoid()
        )

    def encoder(self, x):
        h = self.encoder_conv(x)
        z = self.fc_enc(h)
        return z

    def decoder(self, z):
        h = self.fc_dec(z)
        recon = self.decoder_conv(h)
        return recon

    def forward(self, x):
        z = self.encoder(x); recon = self.decoder(z); return z, recon

# ------------------------
# Resonant Engine (inference-time) — unchanged
# ------------------------
class ResonantEngine:
    def __init__(self, model):
        self.model = model
        self.criterion = nn.MSELoss()

    def resonate(self, x_img, steps=None, lr=None):
        if steps is None: steps = CONFIG['inference_steps']
        if lr is None: lr = CONFIG['inference_lr']

        with torch.no_grad():
            z0 = self.model.encoder(x_img)
            recon0 = self.model.decoder(z0)
            z0_loss = float(self.criterion(recon0, x_img).item())

        z = z0.clone().detach().requires_grad_(True)
        opt = optim.Adam([z], lr=lr)
        loss_val = None
        for _ in range(steps):
            opt.zero_grad()
            recon = self.model.decoder(z)
            loss = self.criterion(recon, x_img)
            loss.backward()
            opt.step()
            loss_val = float(loss.item())
        return z.detach(), loss_val, z0_loss

# ------------------------
# CognitiveMemory: ABS or REL acceptance
# ------------------------
class CognitiveMemory:
    def __init__(self, capacity=5000, abs_thr=0.05, rel_factor=0.9):
        self.capacity = capacity
        self.abs_thr = abs_thr
        self.rel_factor = rel_factor
        self.vectors = deque(maxlen=capacity)
        self.coherences = deque(maxlen=capacity)

    def add_event(self, z_tensor, loss, z0_loss=None):
        store = False
        if loss < self.abs_thr:
            store = True
        elif z0_loss is not None and loss < (self.rel_factor * z0_loss):
            store = True

        if store:
            arr = z_tensor.squeeze(0).cpu().numpy().reshape(-1)
            self.vectors.append(arr)
            self.coherences.append(float(loss))
            return True
        return False

    def to_array(self):
        if len(self.vectors) == 0:
            return np.zeros((0, CONFIG['latent_dim']))
        return np.vstack(list(self.vectors))

    def __len__(self):
        return len(self.vectors)

# ------------------------
# Symbol discovery (kmeans fallback)
# ------------------------
def discover_symbols(memory_array, n_symbols=4):
    if memory_array.shape[0] < n_symbols:
        unique = np.unique(memory_array.round(decimals=8), axis=0)
        labels = np.zeros(memory_array.shape[0], dtype=int)
        for i, xi in enumerate(memory_array):
            labels[i] = int(np.argmin(np.sum((unique - xi) ** 2, axis=1)))
        centers = unique
        return centers, labels

    try:
        kmeans = KMeans(n_clusters=n_symbols, n_init=10, random_state=0)
        labels = kmeans.fit_predict(memory_array)
        centers = kmeans.cluster_centers_
        return centers, labels
    except Exception as e:
        print("[discover_symbols] KMeans failed; using HDBSCAN fallback:", e)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=2)
        labels = clusterer.fit_predict(memory_array)
        unique_labels = [l for l in set(labels) if l != -1]
        centers = []
        for l in unique_labels:
            centers.append(np.mean(memory_array[labels == l], axis=0))
        if len(centers) == 0:
            return np.zeros((0, memory_array.shape[1])), labels
        return np.vstack(centers), labels

# ------------------------
# Graph export (NetworkX -> JSON)
# ------------------------
class SymbolGraph:
    def __init__(self):
        self.G = nx.Graph()
    def add_symbol(self, idx, coherence=0.0):
        self.G.add_node(f"symbol_{idx}", type="symbol", coherence=float(coherence))
    def add_relation(self, a, b, weight=1.0):
        self.G.add_edge(f"symbol_{a}", f"symbol_{b}", weight=float(weight))
    def add_rule(self, rid, desc, conf=1.0):
        self.G.add_node(f"rule_{rid}", type="rule", desc=desc, confidence=float(conf))
    def link_rule_symbol(self, rid, sid):
        self.G.add_edge(f"rule_{rid}", f"symbol_{sid}", weight=1.0)
    def export(self, path):
        nodes=[{**d,'id':n} for n,d in self.G.nodes(data=True)]
        links=[{'source':u,'target':v,**d} for u,v,d in self.G.edges(data=True)]
        with open(path,'w') as f: json.dump({'nodes':nodes,'links':links}, f, indent=2)
        print(f"[SymbolGraph] exported to {path}")

# ------------------------
# MAIN pipeline (improved)
# ------------------------
def main():
    print("=== SRA CORE — improved gating & diagnostics ===")
    env = SnakeEnv(size=CONFIG['grid_size'])
    model = ConvAutoencoder(grid_size=CONFIG['grid_size'], latent_dim=CONFIG['latent_dim']).to(device)

    # --- Pretrain AE (smaller for quick iteration)
    print("[Phase 1] Collecting samples for AE pretraining...")
    samples = []
    while len(samples) < CONFIG['pretrain_samples']:
        obs = env.reset()
        for _ in range(30):
            a = random.randrange(4)
            obs, _, done, _ = env.step(a)
            arr = np.array(obs, dtype=np.float32)
            samples.append(arr.flatten())
            if len(samples) >= CONFIG['pretrain_samples'] or done:
                break
    X = np.vstack(samples)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device).view(-1, 3, CONFIG['grid_size'], CONFIG['grid_size'])

    print(f"[Phase 1] Training AE for {CONFIG['ae_epochs']} epochs on {X_tensor.shape[0]} samples")
    opt_ae = optim.Adam(model.parameters(), lr=CONFIG['ae_lr'])
    for ep in range(1, CONFIG['ae_epochs']+1):
        model.train()
        opt_ae.zero_grad()
        z, recon = model(X_tensor)
        loss = F.mse_loss(recon, X_tensor)
        loss.backward(); opt_ae.step()
        if ep % 2 == 0 or ep == CONFIG['ae_epochs']:
            print(f" AE epoch {ep}/{CONFIG['ae_epochs']} loss={loss.item():.6f}")

    model.eval()
    resonator = ResonantEngine(model)
    memory = CognitiveMemory(capacity=CONFIG['memory_capacity'],
                             abs_thr=CONFIG['memory_abs_threshold'],
                             rel_factor=CONFIG['memory_rel_factor'])

    # --- Phase 2: Exploration with debug prints ---
    print("[Phase 2] Exploration & memory population (improved)")
    for ep in range(1, CONFIG['exploration_episodes'] + 1):
        obs = env.reset()
        ep_reward = 0.0
        steps = 0
        while True:
            # use simple model-based lookahead as before
            snapshot = env.copy_state()
            best_score = -1e9; best_a = 0
            dist_before = env.manhattan_head_to_food()
            for a in range(4):
                sim = SnakeEnv(env.size)
                sim.set_state(snapshot)
                sim_obs, r_sim, done_sim, _ = sim.step(a)
                x_img = obs_to_image_tensor(sim_obs)
                z_opt, loss_opt, z0_loss = resonator.resonate(x_img)
                score = -loss_opt
                if r_sim > 0: score += 10.0
                if done_sim: score -= 10.0
                dist_after = sim.manhattan_head_to_food()
                if dist_before is not None and dist_after is not None and dist_after < dist_before:
                    score += 0.5
                if score > best_score:
                    best_score = score; best_a = a

            action = best_a if random.random() > 0.05 else random.randrange(4)  # small epsilon
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            steps += 1

            # resonance on experienced obs
            x_img = obs_to_image_tensor(obs)
            z_opt, loss_opt, z0_loss = resonator.resonate(x_img)
            stored = memory.add_event(z_opt, loss_opt, z0_loss)

            # debug prints when stored or not
            if stored:
                print(f"  [Stored] ep={ep} step={steps} loss={loss_opt:.4f} z0_loss={z0_loss:.4f} mem_total={len(memory)}")
            else:
                # occasionally print a sample of failed attempts for diagnosis
                if steps % 25 == 0:
                    print(f"  [Skipped] ep={ep} step={steps} loss={loss_opt:.4f} z0_loss={z0_loss:.4f} mem_total={len(memory)}")

            if done or steps > 500:
                break

        print(f"[Explore {ep}] reward={ep_reward:.1f} memories={len(memory)}")

        if len(memory) >= CONFIG['min_memory_for_clustering']:
            print("[Phase 2] collected enough memories; breaking exploration.")
            break

    # --- Phase 3: Clustering ---
    print("[Phase 3] Symbol discovery")
    mem_arr = memory.to_array()
    if mem_arr.shape[0] == 0:
        print("No memories collected — try increasing exploration_episodes or lowering thresholds.")
        return

    centers, labels = None, None
    try:
        centers, labels = discover_symbols(mem_arr, n_symbols=CONFIG['initial_n_symbols'])
        if centers is None or centers.size == 0:
            print("[Phase 3] No centers found.")
        else:
            print(f"[Phase 3] discovered {centers.shape[0]} centers.")
    except Exception as e:
        print("[Phase 3] clustering failed:", e)

    # save artifacts
    np.save(OUTDIR / "memory_vectors.npy", mem_arr)
    if centers is not None and centers.size > 0:
        np.save(OUTDIR / "centers.npy", centers)

    # construct simple symbol graph and export
    sg = SymbolGraph()
    if centers is not None and centers.size > 0:
        mean_coh = float(np.mean(memory.coherences)) if len(memory.coherences) > 0 else 0.0
        for i in range(centers.shape[0]):
            sg.add_symbol(i, coherence=mean_coh)
        for i in range(max(0, centers.shape[0]-1)):
            sg.add_relation(i, i+1)
    sg.add_rule(0, "example_rule", conf=0.95)
    if centers is not None and centers.size > 0:
        sg.link_rule_symbol(0, 0)
    sg.export(str(OUTDIR / "symbol_graph.json"))

    print("=== Completed improved run ===")
    print("Artifacts in:", OUTDIR)

# minimal helper to flatten obs for AE pretraining
def flatten_obs_for_ae(obs):
    arr = np.array(obs, dtype=np.float32)
    H = W = CONFIG['grid_size']
    if arr.shape == (3, H, W):
        return arr.flatten()
    flat = arr.flatten()
    if flat.size != 3*H*W:
        raise RuntimeError(f"Unexpected obs size {flat.size}, expected {3*H*W}")
    return flat

if __name__ == "__main__":
    start = time.time()
    main()
    print("Elapsed:", time.time() - start)

