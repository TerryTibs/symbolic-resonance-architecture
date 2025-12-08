#!/usr/bin/env python3
"""
sra_snake_final.py
Fully-updated SRA Snake system (all patches integrated).

Features:
- AE pretraining (random exploration) with LR scheduler and optional early stopping.
- SRA policy with epsilon-greedy exploration for training.
- Resonant latent optimization with configurable steps and LR.
- Coherence-gated memory insertion with absolute threshold and relative improvement gate.
- Memory capacity / replay guard (FIFO) to avoid runaway memory growth.
- Reward shaping: prefer moves reducing food distance, discourage death.
- Deterministic seeding and reproducibility.
- Robust adaptive clustering, PCA visualization, glyph decoding, artifact saving.
- Debugging prints (z0_loss vs z*_loss distribution).
- Many configuration knobs in CONFIG for easy tuning.

Author: Generated for user
Date: 2025
"""

import os
import random
import time
import math
from pathlib import Path
from collections import deque, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ---------------------------
# CONFIGURATION
# ---------------------------
CONFIG = {
    # === Random / Deterministic ===
    'seed': 42,
    'deterministic': True,

    # === Environment ===
    'env_size': 8,                  # grid size
    'max_episode_steps': 500,       # truncate episodes

    # === Pretraining (random play to collect AE dataset) ===
    'collect_random_states': 2500,
    'random_max_steps': 60,

    # === Autoencoder training ===
    'ae_epochs': 20,
    'ae_batch_size': 128,
    'ae_learning_rate': 1e-3,
    'ae_lr_step': 5,                # reduce LR every N epochs
    'ae_lr_gamma': 0.5,
    'ae_early_stopping_patience': 6, # None to disable

    # === Latent & Resonance ===
    'latent_dim': 16,
    'resonance_steps': 16,          # increased inner-loop steps
    'resonance_lr': 0.12,           # stronger inner-loop lr
    'resonance_use_scheduler': False,

    # === Memory gating / replay ===
    'memory_absolute_threshold': 0.012,   # absolute AE loss threshold (was 0.005)
    'memory_relative_factor': 0.80,       # z* < factor * z0 to allow relative storage
    'memory_capacity': 2000,              # max memories kept (FIFO eviction)

    # === Symbol discovery ===
    'min_memory_for_clustering': 40,
    'min_symbols': 2,
    'max_symbols': 10,

    # === Training episodes (populate memory) ===
    'train_episodes': 60,
    'train_max_steps': 400,

    # === Epsilon schedule ===
    'epsilon_start': 1.0,
    'epsilon_end': 0.02,
    'epsilon_decay': 0.97,  # multiply epsilon per episode

    # === Reward shaping (SRA policy) ===
    'reward_eat': 10.0,        # base reward for eating
    'reward_die': -10.0,       # penalty for death
    'shaping_distance_coeff': 0.5,  # reward for reducing manhattan distance

    # === Output / artifacts ===
    'outdir': 'sra_snake_final_allpatches_out',
    'save_every_n_episodes': 5,

    # === Debug/verbosity ===
    'debug': True,
}

# Derived output dir
OUTDIR = Path(CONFIG['outdir'])
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Deterministic seeding
# ---------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if CONFIG['deterministic']:
        torch.use_deterministic_algorithms(True)

set_seed(CONFIG['seed'])

# ---------------------------
# Helper functions
# ---------------------------
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def flatten_obs(obs):
    """Safe flatten: accepts shape (3,H,W) or (H,W,3) etc and returns 1-D np.float32."""
    a = np.array(obs, dtype=np.float32)
    # normalize channel-first expectation
    if a.ndim == 3 and a.shape[0] == 3:
        return a.flatten()
    elif a.ndim == 3 and a.shape[2] == 3:
        return a.transpose(2,0,1).flatten()
    else:
        return a.flatten()

def obs_to_tensor(obs):
    v = flatten_obs(obs)
    return torch.tensor(v, dtype=torch.float32).unsqueeze(0)

# ---------------------------
# Snake Environment
# ---------------------------
class SnakeEnv:
    def __init__(self, size=8):
        self.size = size
        self.reset()

    def reset(self):
        # initialize snake in center, single cell
        self.snake = [(self.size // 2, self.size // 2)]
        self.dir = (0, 1)  # initially moving right
        self.place_food()
        self.done = False
        self.score = 0
        self.steps = 0
        return self._get_obs()

    def place_food(self):
        empty = [(x, y) for x in range(self.size) for y in range(self.size) if (x,y) not in self.snake]
        if not empty:
            # filled board
            self.food = None
            return
        self.food = random.choice(empty)

    def step(self, action):
        """action: 0=up,1=right,2=down,3=left absolute"""
        dx, dy = [(-1,0),(0,1),(1,0),(0,-1)][action]
        head_x, head_y = self.snake[0]
        nx, ny = head_x + dx, head_y + dy
        self.steps += 1
        # check collision
        if nx < 0 or ny < 0 or nx >= self.size or ny >= self.size or (nx,ny) in self.snake:
            self.done = True
            reward = CONFIG['reward_die']
            return self._get_obs(), reward, True, {}
        # move
        self.snake.insert(0,(nx,ny))
        reward = 0.0
        if self.food is not None and (nx,ny) == self.food:
            reward = CONFIG['reward_eat']
            self.score += 1
            self.place_food()
        else:
            # normal move: drop tail
            self.snake.pop()
        # step termination by max steps (outside)
        return self._get_obs(), reward, False, {}

    def _get_obs(self):
        obs = np.zeros((3, self.size, self.size), dtype=np.float32)
        head = self.snake[0]
        obs[0, head[0], head[1]] = 1.0
        for x,y in self.snake[1:]:
            obs[1, x, y] = 1.0
        if self.food is not None:
            fx,fy = self.food
            obs[2, fx, fy] = 1.0
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

# ---------------------------
# Perceptual Core (Autoencoder)
# ---------------------------
class PerceptualCore(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon

# ---------------------------
# Resonant Engine
# ---------------------------
class ResonantEngine:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.criterion = nn.MSELoss()

    def resonate(self, x, steps=None, lr=None):
        """Perform inference-time optimization on latent z to minimize recon error wrt x.
        Returns (z_opt, final_loss). x is a tensor [1, input_dim].
        """
        if steps is None:
            steps = CONFIG['resonance_steps']
        if lr is None:
            lr = CONFIG['resonance_lr']
        with torch.no_grad():
            z0 = self.model.encoder(x)
            recon0 = self.model.decoder(z0)
            z0_loss = float(self.criterion(recon0, x).item())
        z = z0.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([z], lr=lr)
        loss_val = None
        for _ in range(steps):
            optimizer.zero_grad()
            recon = self.model.decoder(z)
            loss = self.criterion(recon, x)
            loss.backward()
            optimizer.step()
            loss_val = float(loss.item())
        return z.detach(), loss_val, z0_loss

# ---------------------------
# Cognitive Memory with FIFO capacity and gated insertion
# ---------------------------
class CognitiveMemory:
    def __init__(self, capacity=None, abs_threshold=None, rel_factor=None):
        self.vectors = deque(maxlen=capacity if capacity is not None else CONFIG['memory_capacity'])
        self.coherences = deque(maxlen=capacity if capacity is not None else CONFIG['memory_capacity'])
        self.abs_threshold = CONFIG['memory_absolute_threshold'] if abs_threshold is None else abs_threshold
        self.rel_factor = CONFIG['memory_relative_factor'] if rel_factor is None else rel_factor

    def add_event(self, latent_vector, loss, z0_loss=None):
        """Insert latent_vector iff (loss < abs_threshold) OR (loss < rel_factor * z0_loss).
        Returns True if stored.
        """
        store = False
        if loss < self.abs_threshold:
            store = True
        elif (z0_loss is not None) and (loss < (self.rel_factor * z0_loss)):
            store = True
        if store:
            v = latent_vector.detach().cpu().numpy().reshape(-1)
            self.vectors.append(v)
            self.coherences.append(loss)
            return True
        return False

    def to_array(self):
        if len(self.vectors) == 0:
            return np.zeros((0,))
        return np.vstack(list(self.vectors))

    def __len__(self):
        return len(self.vectors)

    def clear(self):
        self.vectors.clear()
        self.coherences.clear()

# ---------------------------
# AE pretraining dataset collection
# ---------------------------
def collect_random_states(env, n_samples=2000, max_steps=60):
    states = []
    for _ in range(n_samples):
        env.reset()
        for _ in range(random.randint(1, max_steps)):
            a = random.randrange(4)
            obs, _, done, _ = env.step(a)
            states.append(flatten_obs(obs))
            if done:
                break
    data = np.stack(states, axis=0)
    return data

# ---------------------------
# AE training
# ---------------------------
def train_autoencoder(core, data, epochs=20, batch_size=128, lr=1e-3, device='cpu',
                      lr_step=5, lr_gamma=0.5, early_stopping_patience=None):
    core.to(device)
    optimizer = optim.Adam(core.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    criterion = nn.MSELoss()

    N = data.shape[0]
    tensor_data = torch.tensor(data, dtype=torch.float32).to(device)

    best_loss = float('inf')
    patience = 0

    for ep in range(1, epochs+1):
        idx = torch.randperm(N)
        total = 0.0
        core.train()
        for i in range(0, N, batch_size):
            batch_idx = idx[i:i+batch_size]
            batch = tensor_data[batch_idx]
            optimizer.zero_grad()
            z, recon = core(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * batch.size(0)
        avg = total / N
        scheduler.step()

        print(f"AE Train Epoch {ep}/{epochs} - Loss {avg:.6f} LR {scheduler.get_last_lr()[0]:.6f}")

        # early stopping
        if early_stopping_patience is not None:
            if avg + 1e-9 < best_loss:
                best_loss = avg
                patience = 0
            else:
                patience += 1
            if patience >= early_stopping_patience:
                print(f"[AE] Early stopping triggered at epoch {ep}. Best loss {best_loss:.6f}")
                break

    core.eval()
    return core

# ---------------------------
# Adaptive symbol discovery (robust)
# ---------------------------
def adaptive_symbol_discovery(memory_vectors, min_k=2, max_k=10):
    """Return (centers, labels, k_found). Guards against duplicate points."""
    if len(memory_vectors) == 0:
        return np.zeros((0,)), np.array([]), 0
    X = np.vstack(memory_vectors)
    N = X.shape[0]
    unique_count = np.unique(X.round(decimals=8), axis=0).shape[0]
    max_k_eff = min(max_k, N)
    min_k_eff = min(min_k, max_k_eff)
    if max_k_eff < min_k_eff:
        # not enough points
        centers = np.unique(X.round(decimals=8), axis=0)
        labels = np.zeros(N, dtype=int)
        for i, xi in enumerate(X):
            diffs = ((centers - xi)**2).sum(axis=1)
            labels[i] = int(np.argmin(diffs))
        return centers, labels, centers.shape[0]
    if unique_count < min_k_eff:
        centers = np.unique(X.round(decimals=8), axis=0)
        labels = np.zeros(N, dtype=int)
        for i, xi in enumerate(X):
            diffs = ((centers - xi)**2).sum(axis=1)
            labels[i] = int(np.argmin(diffs))
        return centers, labels, centers.shape[0]

    ks = list(range(min_k_eff, max_k_eff+1))
    inertias = []
    models = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        km.fit(X)
        inertias.append(km.inertia_)
        models.append(km)
    if len(inertias) == 1:
        best_idx = 0
    else:
        drops = np.diff(inertias)
        rel_drops = drops / (np.array(inertias[:-1]) + 1e-8)
        best_idx = int(np.argmax(np.abs(rel_drops))) + 1
    best_km = models[best_idx]
    centers = best_km.cluster_centers_
    labels = best_km.labels_
    return centers, labels, centers.shape[0]

# ---------------------------
# Reward shaping utilities
# ---------------------------
def manhattan_distance(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

# ---------------------------
# SRA-based action selection (epsilon-greedy)
# ---------------------------
def sra_choose_action(env, core, engine, epsilon=0.0, device='cpu'):
    # epsilon random
    if random.random() < epsilon:
        return random.randrange(4), {'source': 'epsilon_random'}
    base_state = env.copy_state()
    best_score = -1e9
    best_action = 0
    best_details = None
    head_before = env.snake[0]
    dist_before = env.manhattan_head_to_food() if env.food is not None else None

    for a in range(4):
        sim_state = dict(base_state)
        sim = SnakeEnv(env.size)
        sim.set_state(sim_state)
        obs_sim, reward_sim, done_sim, _ = sim.step(a)
        # compute reward shaping: distance delta
        dist_after = sim.manhattan_head_to_food() if sim.food is not None else None
        shaping = 0.0
        if dist_before is not None and dist_after is not None:
            shaping = CONFIG['shaping_distance_coeff'] * (dist_before - dist_after)
        # Now compute resonance coherence to prefer stable, well-explained states
        x = obs_to_tensor(obs_sim).to(device)
        z_opt, loss_opt, z0_loss = engine.resonate(x)
        # score: combine intrinsic (lower loss better) and extrinsic (eat/die/shaping)
        score = -loss_opt + shaping
        if reward_sim > 0:
            score += CONFIG['reward_eat']
        if done_sim:
            score += CONFIG['reward_die']  # heavy penalty
        # small tie-breaker prefer lower loss
        if score > best_score:
            best_score = score
            best_action = a
            best_details = {'sim_reward': reward_sim, 'loss_opt': loss_opt, 'z0_loss': z0_loss, 'shaping': shaping}
    return best_action, best_details

# ---------------------------
# Save artifacts (glyphs, PCA, memory arrays)
# ---------------------------
def save_artifacts(core, centers, memory, labels, tag):
    subdir = OUTDIR / tag
    ensure_dir(subdir)
    np.save(subdir / "memory_vectors.npy", memory.to_array())
    np.save(subdir / "coherences.npy", np.array(memory.coherences))
    if centers is None or centers.size == 0:
        return
    np.save(subdir / "centers.npy", centers)
    # decode centers
    with torch.no_grad():
        for i, c in enumerate(centers):
            z = torch.tensor(c, dtype=torch.float32).unsqueeze(0)
            recon = core.decoder(z).cpu().numpy().reshape(-1)
            try:
                viz = recon.reshape(3, CONFIG['env_size'], CONFIG['env_size']).sum(axis=0)
            except Exception:
                viz = recon.reshape(CONFIG['env_size'], CONFIG['env_size'])
            plt.figure(figsize=(2,2))
            plt.imshow(viz, cmap='gray')
            plt.axis('off')
            plt.title(f"Glyph {i+1}")
            plt.savefig(subdir / f"glyph_{i+1}.png")
            plt.close()
    # PCA
    X = memory.to_array()
    try:
        pca = PCA(n_components=2)
        X2 = pca.fit_transform(X)
        centers2 = pca.transform(centers)
        plt.figure(figsize=(6,5))
        plt.scatter(X2[:,0], X2[:,1], c=labels, cmap='tab10', s=10, alpha=0.6)
        plt.scatter(centers2[:,0], centers2[:,1], c='black', s=80, marker='x')
        plt.title("Memory latent space (PCA)")
        plt.savefig(subdir / "latent_pca.png")
        plt.close()
    except Exception as e:
        print("[save_artifacts] PCA failed:", e)
    # summary
    with open(subdir / "summary.txt", "w") as f:
        f.write(f"memories: {len(memory)}\n")
        f.write(f"clusters: {centers.shape[0]}\n")
        f.write("CONFIG:\n")
        for k,v in CONFIG.items():
            f.write(f"{k}: {v}\n")

# ---------------------------
# Main training + evaluation pipeline
# ---------------------------
def main():
    device = 'cpu'
    env = SnakeEnv(size=CONFIG['env_size'])

    print("=== SRA Snake Final (ALL PATCHES) ===")
    print("Collecting random states for AE pretraining...")
    data = collect_random_states(env, n_samples=CONFIG['collect_random_states'], max_steps=CONFIG['random_max_steps'])
    input_dim = data.shape[1]
    core = PerceptualCore(input_dim=input_dim, latent_dim=CONFIG['latent_dim'])

    print("Training Autoencoder with scheduler and optional early stopping...")
    core = train_autoencoder(core, data,
                             epochs=CONFIG['ae_epochs'],
                             batch_size=CONFIG['ae_batch_size'],
                             lr=CONFIG['ae_learning_rate'],
                             device=device,
                             lr_step=CONFIG['ae_lr_step'],
                             lr_gamma=CONFIG['ae_lr_gamma'],
                             early_stopping_patience=CONFIG['ae_early_stopping_patience'])

    engine = ResonantEngine(core, device=device)
    memory = CognitiveMemory(capacity=CONFIG['memory_capacity'],
                             abs_threshold=CONFIG['memory_absolute_threshold'],
                             rel_factor=CONFIG['memory_relative_factor'])

    # Training episodes with epsilon-greedy exploration to populate memory
    epsilon = CONFIG['epsilon_start']
    print("\n=== Training episodes (population phase) ===")
    for ep in range(1, CONFIG['train_episodes'] + 1):
        obs = env.reset()
        total_reward = 0.0
        steps = 0
        # track some debug stats for losses
        z0_losses = []
        zstar_losses = []
        stored_count = 0
        for t in range(CONFIG['train_max_steps']):
            a, details = sra_choose_action(env, core, engine, epsilon=epsilon, device=device)
            obs, reward, done, _ = env.step(a)
            total_reward += reward
            steps += 1

            # resonance and memory insertion
            x = obs_to_tensor(obs).to(device)
            z_opt, loss_opt, z0_loss = engine.resonate(x, steps=CONFIG['resonance_steps'], lr=CONFIG['resonance_lr'])
            stored = memory.add_event(z_opt, loss_opt, z0_loss)
            if stored:
                stored_count += 1

            z0_losses.append(z0_loss)
            zstar_losses.append(loss_opt)

            if done:
                break

        print(f"[Train {ep}/{CONFIG['train_episodes']}] steps={steps} reward={total_reward} memories={len(memory)} epsilon={epsilon:.3f} stored_this_ep={stored_count}")

        # debug prints showing distribution
        if CONFIG['debug']:
            if len(z0_losses) > 0:
                print(f"  debug z0_loss mean={np.mean(z0_losses):.5f} min={np.min(z0_losses):.5f} z*_mean={np.mean(zstar_losses):.5f} z*_min={np.min(zstar_losses):.5f}")

        # decay epsilon
        epsilon = max(CONFIG['epsilon_end'], epsilon * CONFIG['epsilon_decay'])

        # periodic clustering & artifacts if enough memories
        if len(memory) >= CONFIG['min_memory_for_clustering'] and (ep % CONFIG['save_every_n_episodes'] == 0):
            centers, labels, k = adaptive_symbol_discovery(memory.vectors, min_k=CONFIG['min_symbols'], max_k=CONFIG['max_symbols'])
            print(f"[Train ep {ep}] Adaptive clustering -> {k} symbols discovered.")
            save_artifacts(core, centers, memory, labels, tag=f"train_ep_{ep}")

    # Post-training evaluation (no exploration)
    print("\n=== Evaluation (no exploration) ===")
    for ev in range(1, 9):
        obs = env.reset()
        total_reward = 0.0
        steps = 0
        for t in range(CONFIG['max_episode_steps']):
            a, details = sra_choose_action(env, core, engine, epsilon=0.0, device=device)
            obs, reward, done, _ = env.step(a)
            total_reward += reward
            steps += 1
            if done:
                break
        print(f"[Eval {ev}] steps={steps} reward={total_reward}")

    # Final clustering if enough memories
    if len(memory) >= CONFIG['min_memory_for_clustering']:
        centers, labels, k = adaptive_symbol_discovery(memory.vectors, min_k=CONFIG['min_symbols'], max_k=CONFIG['max_symbols'])
        print(f"[Final] Discovered {k} symbols from {len(memory)} memories.")
        save_artifacts(core, centers, memory, labels, tag="final")
    else:
        print("[Final] Not enough memories for final symbol discovery. Saving partial buffer.")
        np.save(OUTDIR / "memory_vectors_partial.npy", memory.to_array())

    print("\n=== Finished. Artifacts saved to:", OUTDIR)

# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Total time:", time.time() - start_time)
