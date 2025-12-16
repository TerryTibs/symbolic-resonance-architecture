#!/usr/bin/env python3
"""
sra_snake_symbolic_rl.py

Symbolic-Resonance Architecture for Snake + Symbolic RL (DQN + Curiosity)
Combines:
 - Perceptual autoencoder + Resonant inference
 - Coherence-gated memory + adaptive clustering => Symbols
 - Symbol embeddings + resonated latents => state representation
 - DQN policy trained with extrinsic + intrinsic (curiosity) reward
 - Forward model (latent predictor) provides intrinsic reward
 - Artifact saving: glyphs, PCA, memory, models, logs

Author: Generated for user
Date: 2025-12-09
"""
import os, random, time, math
from pathlib import Path
from collections import deque, defaultdict
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
    'deterministic': True,

    # env
    'env_size': 8,
    'train_max_steps': 300,

    # AE pretrain
    'collect_random_states': 2000,
    'random_max_steps': 60,
    'ae_epochs': 12,
    'ae_batch_size': 128,
    'ae_lr': 1e-3,

    # resonance
    'latent_dim': 16,
    'resonance_steps': 8,
    'resonance_lr': 0.06,

    # memory gating
    'memory_abs_threshold': 0.012,
    'memory_rel_factor': 0.80,
    'memory_capacity': 3000,
    'min_memory_for_clustering': 80,

    # clustering
    'min_symbols': 3,
    'max_symbols': 10,

    # RL
    'train_episodes': 120,
    'epsilon_start': 1.0,
    'epsilon_end': 0.02,
    'epsilon_decay': 0.985,
    'replay_capacity': 5000,
    'batch_size': 64,
    'dqn_lr': 1e-3,
    'gamma': 0.99,
    'target_update_every': 200,  # steps

    # curiosity
    'curiosity_beta': 1.0,  # weight of intrinsic reward
    'forward_lr': 1e-3,

    # artifact/output
    'outdir': 'sra_snake_symbolic_rl_out',
    'save_every_epochs': 20,

    'debug': True,
    'device': 'cpu',  # set 'cuda' if available
}

OUTDIR = Path(CONFIG['outdir'])
OUTDIR.mkdir(parents=True, exist_ok=True)

# deterministic
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if CONFIG['deterministic']:
        torch.use_deterministic_algorithms(True)

set_seed(CONFIG['seed'])
device = torch.device(CONFIG['device'])

# -------------------------
# Minimal Snake env (same as earlier)
# -------------------------
class SnakeEnv:
    def __init__(self, size=8):
        self.size = size
        self.reset()
    def reset(self):
        self.snake = [(self.size//2, self.size//2)]
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
        head_x,head_y = self.snake[0]
        nx,ny = head_x + dx, head_y + dy
        self.steps += 1
        if nx<0 or ny<0 or nx>=self.size or ny>=self.size or (nx,ny) in self.snake:
            self.done=True
            return self._get_obs(), -10.0, True, {}
        self.snake.insert(0,(nx,ny))
        reward=0.0
        if self.food and (nx,ny)==self.food:
            reward=10.0
            self.score+=1
            self.place_food()
        else:
            self.snake.pop()
        return self._get_obs(), reward, False, {}
    def _get_obs(self):
        obs = np.zeros((3,self.size,self.size),dtype=np.float32)
        hx,hy = self.snake[0]
        obs[0,hx,hy]=1.0
        for x,y in self.snake[1:]:
            obs[1,x,y]=1.0
        if self.food:
            fx,fy = self.food
            obs[2,fx,fy]=1.0
        return obs
    def copy_state(self):
        return {'snake': list(self.snake), 'food': self.food, 'done': self.done}
    def set_state(self, st):
        self.snake = list(st['snake']); self.food = st['food']; self.done = st['done']
    def manhattan_head_to_food(self):
        if not self.food: return None
        hx,hy = self.snake[0]; fx,fy = self.food
        return abs(hx-fx)+abs(hy-fy)

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
# Perceptual core (AE)
# -------------------------
class PerceptualCore(nn.Module):
    def __init__(self, input_dim, latent=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim,256), nn.ReLU(),
            nn.Linear(256,128), nn.ReLU(),
            nn.Linear(128, latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent,128), nn.ReLU(),
            nn.Linear(128,256), nn.ReLU(),
            nn.Linear(256,input_dim), nn.Sigmoid()
        )
    def forward(self,x):
        z = self.encoder(x)
        return z, self.decoder(z)

# -------------------------
# Resonant engine (optimise z to explain x)
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
        z = z0.clone().detach().requires_grad_(True)
        opt = optim.Adam([z], lr=lr)
        loss_val = None
        for _ in range(steps):
            opt.zero_grad()
            recon = self.model.decoder(z)
            loss = self.criterion(recon, x)
            loss.backward()
            opt.step()
            loss_val = float(loss.item())
        return z.detach(), loss_val, z0_loss

# -------------------------
# Memory (FIFO + gating)
# -------------------------
class CognitiveMemory:
    def __init__(self, capacity=3000, abs_thr=0.012, rel_factor=0.8):
        self.vectors = deque(maxlen=capacity)
        self.coherences = deque(maxlen=capacity)
        self.abs_thr = abs_thr; self.rel_factor = rel_factor
    def add_event(self, latent_vector, loss, z0_loss=None):
        store=False
        if loss < self.abs_thr: store=True
        elif z0_loss is not None and loss < self.rel_factor * z0_loss: store=True
        if store:
            self.vectors.append(latent_vector.detach().cpu().numpy().reshape(-1))
            self.coherences.append(loss)
            return True
        return False
    def to_array(self):
        if len(self.vectors)==0: return np.zeros((0,))
        return np.vstack(list(self.vectors))
    def __len__(self): return len(self.vectors)
    def clear(self): self.vectors.clear(); self.coherences.clear()

# -------------------------
# Adaptive clustering -> symbols
# -------------------------
def adaptive_symbol_discovery(mem_vectors, min_k=3, max_k=10):
    if len(mem_vectors)==0: return np.zeros((0,)), np.array([]), 0
    X = np.vstack(mem_vectors); N = X.shape[0]
    unique_count = np.unique(X.round(decimals=8), axis=0).shape[0]
    max_k_eff = min(max_k, N); min_k_eff = min(min_k, max_k_eff)
    if max_k_eff < min_k_eff:
        centers = np.unique(X.round(decimals=8), axis=0)
        labels = np.zeros(N,dtype=int)
        for i,xi in enumerate(X):
            labels[i] = int(np.argmin(((centers-xi)**2).sum(axis=1)))
        return centers, labels, centers.shape[0]
    if unique_count < min_k_eff:
        centers = np.unique(X.round(decimals=8), axis=0)
        labels = np.zeros(N,dtype=int)
        for i,xi in enumerate(X):
            labels[i] = int(np.argmin(((centers-xi)**2).sum(axis=1)))
        return centers, labels, centers.shape[0]
    ks = list(range(min_k_eff, max_k_eff+1)); inertias=[]; models=[]
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        inertias.append(km.inertia_); models.append(km)
    if len(inertias)==1: best_idx=0
    else:
        drops = np.diff(inertias); rel = drops/(np.array(inertias[:-1])+1e-8)
        best_idx = int(np.argmax(np.abs(rel)))+1
    best_km = models[best_idx]
    return best_km.cluster_centers_, best_km.labels_, best_km.cluster_centers_.shape[0]

# -------------------------
# DQN (simple MLP) and replay
# -------------------------
class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.buf = deque(maxlen=capacity)
    def push(self, s, a, r, s2, done):
        self.buf.append((s,a,r,s2,done))
    def sample(self, batch_size):
        idx = np.random.choice(len(self.buf), batch_size, replace=False)
        batch = [self.buf[i] for i in idx]
        s,a,r,s2,d = zip(*batch)
        return np.vstack(s), np.array(a), np.array(r), np.vstack(s2), np.array(d)
    def __len__(self): return len(self.buf)

class DQN(nn.Module):
    def __init__(self, input_dim, n_actions=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU(),
            nn.Linear(128,n_actions)
        )
    def forward(self,x): return self.net(x)

# -------------------------
# Forward model for curiosity
# -------------------------
class ForwardModel(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # predict next latent from (latent, action embedding)
        self.net = nn.Sequential(
            nn.Linear(input_dim,128), nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
    def forward(self,x): return self.net(x)

# -------------------------
# Utility: get symbol id and embed
# -------------------------
def assign_symbols(centers, z_vecs):
    # centers: k x latent_dim, z_vecs: (n,latent_dim)
    if centers.size==0: return np.zeros(len(z_vecs), dtype=int)
    diffs = ((z_vecs[:,None,:] - centers[None,:,:])**2).sum(axis=2)  # n x k
    return np.argmin(diffs, axis=1)

# -------------------------
# Save utilities
# -------------------------
def save_glyphs(core, centers, tag):
    sub = OUTDIR / tag; sub.mkdir(parents=True, exist_ok=True)
    np.save(sub / "centers.npy", centers)
    with torch.no_grad():
        for i,c in enumerate(centers):
            z = torch.tensor(c, dtype=torch.float32).unsqueeze(0).to(device)
            recon = core.decoder(z).cpu().numpy().reshape(-1)
            try:
                viz = recon.reshape(3, CONFIG['env_size'], CONFIG['env_size']).sum(axis=0)
            except:
                viz = recon.reshape(CONFIG['env_size'], CONFIG['env_size'])
            plt.figure(figsize=(2,2)); plt.imshow(viz, cmap='gray'); plt.axis('off')
            plt.savefig(sub / f"glyph_{i+1}.png"); plt.close()
    return sub

def save_memory_pca(memory, labels, tag):
    sub = OUTDIR / tag; sub.mkdir(parents=True, exist_ok=True)
    X = memory.to_array()
    np.save(sub / "memory.npy", X); np.save(sub / "labels.npy", labels)
    try:
        pca = PCA(n_components=2); X2 = pca.fit_transform(X)
        plt.figure(figsize=(6,5)); plt.scatter(X2[:,0], X2[:,1], c=labels, s=8, cmap='tab10'); plt.title("Memory PCA")
        plt.savefig(sub / "memory_pca.png"); plt.close()
    except Exception as e:
        print("PCA failed:", e)

# -------------------------
# Main pipeline
# -------------------------
def main():
    env = SnakeEnv(size=CONFIG['env_size'])
    print("1) Collect random states to pretrain AE")
    data = []
    for _ in range(CONFIG['collect_random_states']):
        obs = env.reset()
        for _ in range(random.randint(1, CONFIG['random_max_steps'])):
            a = random.randrange(4)
            obs,_,done,_ = env.step(a)
            data.append(flatten_obs(obs))
            if done: break
    data = np.vstack(data); input_dim = data.shape[1]
    core = PerceptualCore(input_dim, latent=CONFIG['latent_dim']).to(device)
    # AE training
    optimizer = optim.Adam(core.parameters(), lr=CONFIG['ae_lr'])
    criterion = nn.MSELoss()
    print("Training AE...")
    N = data.shape[0]; tensor_data = torch.tensor(data, dtype=torch.float32).to(device)
    for ep in range(CONFIG['ae_epochs']):
        perm = torch.randperm(N)
        total=0.0
        core.train()
        for i in range(0,N, CONFIG['ae_batch_size']):
            idx = perm[i:i+CONFIG['ae_batch_size']]
            batch = tensor_data[idx]
            optimizer.zero_grad()
            z,recon = core(batch)
            loss = criterion(recon, batch)
            loss.backward(); optimizer.step()
            total += float(loss.item())*batch.size(0)
        avg = total/N
        print(f" AE Epoch {ep+1}/{CONFIG['ae_epochs']} loss {avg:.6f}")
    core.eval()
    engine = ResonantEngine(core)
    memory = CognitiveMemory(capacity=CONFIG['memory_capacity'],
                             abs_thr=CONFIG['memory_abs_threshold'],
                             rel_factor=CONFIG['memory_rel_factor'])
    # populate memory via SRA policy with epsilon
    print("2) Populate memory via exploration")
    epsilon = CONFIG['epsilon_start']
    for ep in range(1, CONFIG['train_episodes']+1):
        obs = env.reset(); total_reward=0.0
        for t in range(CONFIG['train_max_steps']):
            # epsilon random or SRA choose
            if random.random() < epsilon:
                a = random.randrange(4); details={'source':'eps'}
            else:
                # sra: simulate actions and prefer low loss and shaping
                base_state = env.copy_state(); best_score=-1e9; best_a=0; best_details=None
                dist_before = env.manhattan_head_to_food()
                for a_try in range(4):
                    sim = SnakeEnv(env.size); sim.set_state(base_state)
                    obs_sim, r_sim, done_sim, _ = sim.step(a_try)
                    dist_after = sim.manhattan_head_to_food()
                    shaping = 0.5 * (dist_before - dist_after) if (dist_before is not None and dist_after is not None) else 0.0
                    x = obs_to_tensor(obs_sim)
                    z_opt, loss_opt, z0_loss = engine.resonate(x)
                    score = -loss_opt + shaping
                    if r_sim>0: score += 10.0
                    if done_sim: score -= 10.0
                    if score > best_score:
                        best_score=score; best_a=a_try; best_details={'r_sim':r_sim,'loss':loss_opt}
                a, details = best_a, best_details
            obs, reward, done, _ = env.step(a)
            total_reward += reward
            # resonance on true obs and memory insert
            x = obs_to_tensor(obs)
            z_opt, loss_opt, z0_loss = engine.resonate(x)
            stored = memory.add_event(z_opt, loss_opt, z0_loss)
            if stored and CONFIG['debug']: pass
            if done: break
        epsilon = max(CONFIG['epsilon_end'], epsilon * CONFIG['epsilon_decay'])
        print(f"[Exploit Ep {ep}] reward={total_reward:.2f} mem={len(memory)} eps={epsilon:.3f}")
        if len(memory) >= CONFIG['min_memory_for_clustering'] and ep % CONFIG['save_every_epochs']==0:
            centers, labels, k = adaptive_symbol_discovery(memory.vectors, min_k=CONFIG['min_symbols'], max_k=CONFIG['max_symbols'])
            print(f"  -> discovered {k} symbols")
            save_glyphs(core, centers, tag=f"symbols_ep{ep}")
            save_memory_pca(memory, labels, tag=f"mem_ep{ep}")
    # After exploration, cluster to get symbols
    if len(memory) < CONFIG['min_memory_for_clustering']:
        print("Not enough memories; exiting. Try increasing exploration.")
        return
    centers, labels, k = adaptive_symbol_discovery(memory.vectors, min_k=CONFIG['min_symbols'], max_k=CONFIG['max_symbols'])
    print("Final clustering -> k =", k)
    save_glyphs(core, centers, tag="symbols_final")
    save_memory_pca(memory, labels, tag="mem_final")
    # Create symbol embedding (learned)
    symbol_k = centers.shape[0]
    symbol_embed_dim = 8
    symbol_embedding = nn.Embedding(symbol_k, symbol_embed_dim).to(device)
    # DQN input: [latent_dim + symbol_embed_dim]
    dqn_input_dim = CONFIG['latent_dim'] + symbol_embed_dim
    policy = DQN(dqn_input_dim, n_actions=4).to(device)
    target = DQN(dqn_input_dim, n_actions=4).to(device)
    target.load_state_dict(policy.state_dict())
    dqn_opt = optim.Adam(policy.parameters(), lr=CONFIG['dqn_lr'])
    # forward model: predicts next latent from [latent + action_onehot]
    forward_input_dim = CONFIG['latent_dim'] + 4
    forward_model = ForwardModel(forward_input_dim, CONFIG['latent_dim']).to(device)
    fwd_opt = optim.Adam(forward_model.parameters(), lr=CONFIG['forward_lr'])
    replay = ReplayBuffer(capacity=CONFIG['replay_capacity'])
    step_count = 0
    print("3) Training policy (DQN) with intrinsic curiosity")
    for ep in range(1, CONFIG['train_episodes']+1):
        obs = env.reset(); total_reward=0.0; total_intrinsic=0.0
        for t in range(CONFIG['train_max_steps']):
            # get resonated latent for observation
            x = obs_to_tensor(obs)
            z_opt, loss_opt, z0_loss = engine.resonate(x)
            z_vec = z_opt.cpu().numpy().reshape(-1)
            # find symbol id
            sym_id = int(np.argmin(((centers - z_vec)**2).sum(axis=1)))
            sym_tensor = torch.tensor([sym_id], dtype=torch.long).to(device)
            sym_emb = symbol_embedding(sym_tensor).squeeze(0)
            state = torch.cat([z_opt.squeeze(0), sym_emb], dim=0).unsqueeze(0)  # 1 x D
            # epsilon greedy (reuse schedule)
            eps = max(CONFIG['epsilon_end'], CONFIG['epsilon_start'] * (CONFIG['epsilon_decay']**ep))
            if random.random() < eps:
                action = random.randrange(4)
            else:
                with torch.no_grad():
                    qvals = policy(state.to(device))
                    action = int(torch.argmax(qvals).item())
            # step
            obs2, ext_reward, done, _ = env.step(action)
            # compute intrinsic reward via forward model prediction error
            z2_opt, loss2_opt, z2_0 = engine.resonate(obs_to_tensor(obs2))
            z2 = z2_opt.squeeze(0)
            # prepare forward input: [z_opt, action_onehot]
            a_onehot = torch.zeros(4, device=device); a_onehot[action]=1.0
            fwd_in = torch.cat([z_opt.squeeze(0), a_onehot], dim=0).unsqueeze(0)
            pred_z2 = forward_model(fwd_in)
            intrinsic = float(((pred_z2.squeeze(0) - z2)**2).mean().item())
            total_intrinsic += intrinsic
            # total reward
            reward = ext_reward + CONFIG['curiosity_beta'] * intrinsic
            total_reward += ext_reward
            # push to replay (state vectors numeric)
            s_np = np.hstack([z_vec, symbol_embedding(sym_tensor).detach().cpu().numpy().reshape(-1)])
            # compute next state's symbol (for s2) using centers
            z2_np = z2.detach().cpu().numpy().reshape(-1)
            sym2_id = int(np.argmin(((centers - z2_np)**2).sum(axis=1)))
            s2_np = np.hstack([z2_np, symbol_embedding(torch.tensor([sym2_id],dtype=torch.long).to(device)).detach().cpu().numpy().reshape(-1)])
            replay.push(s_np, action, reward, s2_np, done)
            # train forward model (supervised)
            fwd_loss = ((pred_z2 - z2.detach().unsqueeze(0))**2).mean()
            fwd_opt.zero_grad(); fwd_loss.backward(); fwd_opt.step()
            # sample DQN
            if len(replay) >= CONFIG['batch_size']:
                sb, ab, rb, s2b, db = replay.sample(CONFIG['batch_size'])
                sb_t = torch.tensor(sb, dtype=torch.float32).to(device)
                ab_t = torch.tensor(ab, dtype=torch.long).to(device)
                rb_t = torch.tensor(rb, dtype=torch.float32).to(device)
                s2b_t = torch.tensor(s2b, dtype=torch.float32).to(device)
                db_t = torch.tensor(db, dtype=torch.float32).to(device)
                qvals = policy(sb_t).gather(1, ab_t.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    qnext = target(s2b_t).max(1)[0]
                    qtarget = rb_t + CONFIG['gamma'] * qnext * (1.0 - db_t)
                loss_dqn = ((qvals - qtarget)**2).mean()
                dqn_opt.zero_grad(); loss_dqn.backward(); dqn_opt.step()
            # update target
            step_count += 1
            if step_count % CONFIG['target_update_every'] == 0:
                target.load_state_dict(policy.state_dict())
            obs = obs2
            if done: break
        if ep % CONFIG['save_every_epochs']==0:
            torch.save(policy.state_dict(), OUTDIR/f"policy_ep{ep}.pth")
            torch.save(forward_model.state_dict(), OUTDIR/f"forward_ep{ep}.pth")
            np.save(OUTDIR/f"memory_ep{ep}.npy", memory.to_array())
        print(f"[Policy Ep {ep}] ext_reward={total_reward:.2f} intrinsic_sum={total_intrinsic:.4f} replay={len(replay)}")
    # final artifacts
    torch.save(policy.state_dict(), OUTDIR/"policy_final.pth")
    torch.save(forward_model.state_dict(), OUTDIR/"forward_final.pth")
    np.save(OUTDIR/"memory_final.npy", memory.to_array())
    save_glyphs(core, centers, tag="symbols_final")
    save_memory_pca(memory, labels, tag="mem_final")
    print("DONE. Artifacts:", OUTDIR)

if __name__ == "__main__":
    t0=time.time()
    main()
    print("Elapsed:", time.time()-t0)

