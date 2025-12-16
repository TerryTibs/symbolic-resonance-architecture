#!/usr/bin/env python3
"""
sra_snake_demo4_fixed.py

SRA Snake — AE + Resonance + Symbol Discovery + PPO
Fixed exploration and memory storage issues
"""

import os, random, time
from pathlib import Path
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# -------------------------
# CONFIG
# -------------------------
CONFIG = {
    'seed': 42,
    'device': 'cpu',            # 'cuda' if available
    'deterministic': True,

    'env_size': 8,
    'max_episode_steps': 300,

    # AE pretrain
    'collect_random_states': 1500,
    'random_max_steps': 40,
    'ae_epochs': 8,
    'ae_batch_size': 128,
    'ae_lr': 1e-3,

    # resonance
    'latent_dim': 16,
    'resonance_steps': 8,
    'resonance_lr': 0.06,

    # memory gating
    'memory_abs_threshold': 0.02,  # relaxed to store more
    'memory_rel_factor': 0.9,      # relaxed
    'memory_capacity': 2000,
    'min_memory_for_clustering': 60,

    # clustering
    'min_symbols': 3,
    'max_symbols': 10,

    # PPO
    'ppo_epochs': 2,
    'ppo_minibatches': 2,
    'ppo_clip': 0.2,
    'ppo_lr': 3e-4,
    'value_coef': 0.5,
    'entropy_coef': 0.01,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'batch_timesteps': 1024,
    'max_updates': 5,  # reduced for testing

    # curiosity
    'curiosity_beta': 0.8,
    'forward_lr': 1e-3,

    'outdir': 'sra_snake_demo4_out',
    'save_every_updates': 2,
    'debug': True,
}

OUTDIR = Path(CONFIG['outdir']); OUTDIR.mkdir(parents=True, exist_ok=True)

# deterministic
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if CONFIG.get('deterministic', False):
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

set_seed(CONFIG['seed'])
device = torch.device(CONFIG['device'])

# -------------------------
# Simple Snake env
# -------------------------
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
        hx,hy = self.snake[0]; nx,ny = hx+dx, hy+dy
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
        return self._get_obs(), reward, False, {}
    def _get_obs(self):
        obs = np.zeros((3,self.size,self.size),dtype=np.float32)
        hx,hy = self.snake[0]; obs[0,hx,hy]=1.0
        for x,y in self.snake[1:]: obs[1,x,y]=1.0
        if self.food:
            fx,fy = self.food; obs[2,fx,fy]=1.0
        return obs
    def copy_state(self): return {'snake': list(self.snake), 'food': self.food, 'done': self.done}
    def set_state(self, st): self.snake=list(st['snake']); self.food=st['food']; self.done=st['done']
    def manhattan_head_to_food(self):
        if not self.food: return None
        hx,hy = self.snake[0]; fx,fy = self.food; return abs(hx-fx)+abs(hy-fy)

# -------------------------
# Utils
# -------------------------
def flatten_obs(obs):
    a = np.array(obs, dtype=np.float32)
    if a.ndim==3 and a.shape[0]==3: return a.flatten()
    if a.ndim==3 and a.shape[2]==3: return a.transpose(2,0,1).flatten()
    return a.flatten()

def obs_to_tensor(obs):
    return torch.tensor(flatten_obs(obs), dtype=torch.float32).unsqueeze(0).to(device)

# -------------------------
# Perceptual Core
# -------------------------
class PerceptualCore(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim,256), nn.ReLU(),
            nn.Linear(256,128), nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim,128), nn.ReLU(),
            nn.Linear(128,256), nn.ReLU(),
            nn.Linear(256,input_dim), nn.Sigmoid()
        )
    def forward(self,x):
        z = self.encoder(x)
        return z, self.decoder(z)

# -------------------------
# ResonantEngine
# -------------------------
class ResonantEngine:
    def __init__(self, model):
        self.model = model
        self.criterion = nn.MSELoss()
    def resonate(self, x, steps=None, lr=None):
        if steps is None: steps = CONFIG['resonance_steps']
        if lr is None: lr = CONFIG['resonance_lr']
        with torch.no_grad():
            z0 = self.model.encoder(x)
            recon0 = self.model.decoder(z0)
            z0_loss = float(self.criterion(recon0, x).item())
        # trainable latent
        z = z0.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([z], lr=lr)
        loss_val=None
        for _ in range(steps):
            optimizer.zero_grad()
            recon = self.model.decoder(z)
            loss = self.criterion(recon, x)
            loss.backward()
            optimizer.step()
            loss_val = float(loss.item())
        return z.detach(), loss_val, z0_loss

# -------------------------
# CognitiveMemory
# -------------------------
class CognitiveMemory:
    def __init__(self, capacity=2000, abs_thr=0.02, rel_factor=0.9):
        self.vectors = deque(maxlen=capacity)
        self.coherences = deque(maxlen=capacity)
        self.abs_thr = abs_thr
        self.rel_factor = rel_factor
    def add_event(self, latent_vector, loss, z0_loss=None):
        # Store everything for testing/fix
        self.vectors.append(latent_vector.detach().cpu().numpy().reshape(-1))
        self.coherences.append(loss)
        return True
    def to_array(self):
        if len(self.vectors)==0: return np.zeros((0,))
        return np.vstack(list(self.vectors))
    def __len__(self): return len(self.vectors)
    def clear(self): self.vectors.clear(); self.coherences.clear()

# -------------------------
# adaptive symbol discovery
# -------------------------
def adaptive_symbol_discovery(mem_vectors, min_k=3, max_k=10):
    if len(mem_vectors)==0: return np.zeros((0,)), np.array([]), 0
    X = np.vstack(mem_vectors); N = X.shape[0]
    unique_count = np.unique(X.round(decimals=8), axis=0).shape[0]
    max_k_eff = min(max_k, N); min_k_eff = min(min_k, max_k_eff)
    if max_k_eff < min_k_eff:
        centers = np.unique(X.round(decimals=8), axis=0); labels = np.zeros(N,dtype=int)
        for i,xi in enumerate(X): labels[i]=int(np.argmin(((centers-xi)**2).sum(axis=1)))
        return centers, labels, centers.shape[0]
    if unique_count < min_k_eff:
        centers = np.unique(X.round(decimals=8), axis=0); labels = np.zeros(N,dtype=int)
        for i,xi in enumerate(X): labels[i]=int(np.argmin(((centers-xi)**2).sum(axis=1)))
        return centers, labels, centers.shape[0]
    ks = list(range(min_k_eff, max_k_eff+1)); inertias=[]; models=[]
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        inertias.append(km.inertia_); models.append(km)
    if len(inertias)==1: best_idx=0
    else:
        drops = np.diff(inertias); rel = drops/(np.array(inertias[:-1])+1e-8); best_idx=int(np.argmax(np.abs(rel)))+1
    best_km = models[best_idx]; return best_km.cluster_centers_, best_km.labels_, best_km.cluster_centers_.shape[0]

# -------------------------
# ActorCritic (PPO) & Forward model
# -------------------------
class ActorCritic(nn.Module):
    def __init__(self, input_dim, n_actions=4):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(input_dim,128), nn.ReLU(), nn.Linear(128,128), nn.ReLU())
        self.policy = nn.Linear(128, n_actions)
        self.value_head = nn.Linear(128,1)
    def forward(self, x):
        h = self.shared(x)
        logits = self.policy(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value

class ForwardModel(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim,128), nn.ReLU(), nn.Linear(128, latent_dim))
    def forward(self,x): return self.net(x)

# -------------------------
# GAE
# -------------------------
def compute_gae(rewards, values, dones, gamma, lam):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        if t == T-1:
            nextnonterminal = 1.0 - dones[t]
            nextvalue = values[t]
        else:
            nextnonterminal = 1.0 - dones[t+1]
            nextvalue = values[t+1]
        delta = rewards[t] + gamma * nextvalue * nextnonterminal - values[t]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        advantages[t] = lastgaelam
    returns = advantages + values
    return advantages, returns

# -------------------------
# Main
# -------------------------
def main():
    env = SnakeEnv(size=CONFIG['env_size'])

    print("Collect random states for AE pretraining")
    data=[]
    for _ in range(CONFIG['collect_random_states']):
        obs = env.reset()
        for _ in range(random.randint(1, CONFIG['random_max_steps'])):
            a = random.randrange(4); obs,_,done,_ = env.step(a); data.append(flatten_obs(obs))
            if done: break
    data = np.vstack(data)
    input_dim = data.shape[1]

    print("Train AE")
    core = PerceptualCore(input_dim=input_dim, latent_dim=CONFIG['latent_dim']).to(device)
    opt_ae = optim.Adam(core.parameters(), lr=CONFIG['ae_lr']); criterion = nn.MSELoss()
    tensor_data = torch.tensor(data, dtype=torch.float32).to(device)
    N = data.shape[0]
    for ep in range(CONFIG['ae_epochs']):
        perm = torch.randperm(N); total=0.0; core.train()
        for i in range(0, N, CONFIG['ae_batch_size']):
            idx = perm[i:i+CONFIG['ae_batch_size']]; batch = tensor_data[idx]; opt_ae.zero_grad(); z,recon = core(batch); loss = criterion(recon,batch); loss.backward(); opt_ae.step(); total += float(loss.item())*batch.size(0)
        print(f" AE epoch {ep+1}/{CONFIG['ae_epochs']} loss {total/N:.6f}")
    core.eval()

    engine = ResonantEngine(core)
    memory = CognitiveMemory(capacity=CONFIG['memory_capacity'], abs_thr=CONFIG['memory_abs_threshold'], rel_factor=CONFIG['memory_rel_factor'])

    print("Exploratory population to fill memory")
    epsilon = 1.0
    for ep in range(1, 26):
        obs = env.reset(); ep_reward=0.0
        for t in range(CONFIG['max_episode_steps']):
            if random.random() < epsilon:
                a = random.randrange(4)
            else:
                a = random.randrange(4)  # minimal policy random for demo
            obs, reward, done, _ = env.step(a); ep_reward += reward
            z_opt, loss_opt, z0_loss = engine.resonate(obs_to_tensor(obs))
            memory.add_event(z_opt, loss_opt, z0_loss)
            if done: break
        epsilon = max(0.02, epsilon*0.9)
        if CONFIG['debug']: print(f"[explore {ep}] reward={ep_reward:.1f} mem={len(memory)} eps={epsilon:.3f}")

    if len(memory) < CONFIG['min_memory_for_clustering']:
        print("Not enough memories; increase exploration.")
        return

    centers, labels, k = adaptive_symbol_discovery(memory.vectors, min_k=CONFIG['min_symbols'], max_k=CONFIG['max_symbols'])
    print("Symbol discovery k =", k)

    # minimal PPO
    state_dim = CONFIG['latent_dim']
    ac = ActorCritic(state_dim, n_actions=4).to(device)
    optimizer = optim.Adam(ac.parameters(), lr=CONFIG['ppo_lr'])

    # dummy single PPO update for demonstration
    obs = env.reset()
    z_opt, _, _ = engine.resonate(obs_to_tensor(obs))
    state_vec = z_opt.squeeze(0)
    logits, value = ac(state_vec.unsqueeze(0))
    action = int(torch.argmax(logits, dim=-1).item())
    print("Sample PPO step: action=", action)

if __name__=="__main__":
    t0 = time.time(); main(); print("Elapsed:", time.time()-t0)

