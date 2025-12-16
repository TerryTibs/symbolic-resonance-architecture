#!/usr/bin/env python3
"""
SRA — Full RL + Symbolic Hybrid (Unified single-file script)

Features:
- Conv Autoencoder pretraining
- Resonant inference-time latent optimization
- Coherence-gated memory
- Symbol discovery (KMeans)
- Symbol embedding
- PPO training with intrinsic curiosity (forward predictor)
- Save artifacts and plots

Run:
    python sra_full_rl_unified.py

Author: assistant (for user)
Date: 2025-12-10
"""
import os
import random
import time
from pathlib import Path
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.cluster import KMeans

# -------------------------
# CONFIG
# -------------------------
CONFIG = {
    # device
    'device': 'cpu',  # 'cuda' if you have GPU and torch.cuda.is_available()

    # environment
    'grid_size': 8,
    'max_episode_steps': 300,

    # AE pretraining
    'pretrain_samples': 1500,
    'ae_epochs': 8,
    'ae_batch_size': 128,
    'ae_lr': 1e-3,

    # latent/resonance
    'latent_dim': 16,
    'inference_steps': 10,
    'inference_lr': 0.06,

    # memory gating (abs OR relative)
    'memory_abs_threshold': 0.05,
    'memory_rel_factor': 0.90,
    'memory_capacity': 2000,
    'min_memory_for_clustering': 100,

    # symbolic discovery
    'n_symbols': 6,

    # PPO / RL
    'ppo_updates': 80,
    'rollout_steps': 512,
    'ppo_epochs': 4,
    'ppo_minibatches': 4,
    'ppo_lr': 2.5e-4,
    'ppo_clip': 0.2,
    'value_coef': 0.5,
    'entropy_coef': 0.01,
    'gamma': 0.99,
    'gae_lambda': 0.95,

    # curiosity (forward predictor)
    'curiosity_beta': 0.8,
    'forward_lr': 1e-3,

    # exploration phase (to populate memory)
    'explore_episodes': 40,

    # output
    'outdir': 'sra_unified_out',
    'seed': 42,
    'debug': True,
}

OUTDIR = Path(CONFIG['outdir']); OUTDIR.mkdir(parents=True, exist_ok=True)
device = torch.device(CONFIG['device'])

# reproducibility
random.seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])

# -------------------------
# Simple Snake Environment
# -------------------------
class SnakeEnv:
    def __init__(self, size=8):
        self.size = size
        self.reset()
    def reset(self):
        # snake as list of (x,y), head at index 0
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
        self.snake.insert(0, (nx,ny))
        reward=0.0
        if self.food is not None and (nx,ny) == self.food:
            reward = 10.0
            self.score += 1
            self.place_food()
        else:
            self.snake.pop()
        if self.steps >= CONFIG['max_episode_steps']:
            self.done = True
        return self._get_obs(), reward, self.done, {}
    def _get_obs(self):
        obs = np.zeros((3, self.size, self.size), dtype=np.float32)
        hx,hy = self.snake[0]; obs[0,hx,hy] = 1.0
        for x,y in self.snake[1:]: obs[1,x,y] = 1.0
        if self.food is not None:
            fx,fy = self.food; obs[2,fx,fy] = 1.0
        return obs
    def copy_state(self):
        return {'snake': list(self.snake), 'food': self.food, 'done': self.done}
    def set_state(self, st):
        self.snake = list(st['snake']); self.food = st['food']; self.done = st['done']
    def manhattan_head_to_food(self):
        if self.food is None: return None
        hx,hy = self.snake[0]; fx,fy = self.food
        return abs(hx-fx)+abs(hy-fy)

# -------------------------
# Helpers: obs conversion
# -------------------------
def obs_to_image_tensor(obs):
    """
    Convert (3,H,W) numpy -> torch tensor (1,3,H,W) on device
    """
    arr = np.array(obs, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError("obs must be shape (3,H,W)")
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(device)

def flatten_obs_for_ae(obs):
    return np.array(obs, dtype=np.float32).flatten()

# -------------------------
# Conv Autoencoder (Perceptual Core)
# -------------------------
class ConvAutoencoder(nn.Module):
    def __init__(self, grid_size=8, latent_dim=16):
        super().__init__()
        C=3; H=grid_size; W=grid_size
        self.enc = nn.Sequential(
            nn.Conv2d(C, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        flat_dim = 32 * H * W
        self.fc_enc = nn.Linear(flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat_dim)
        self.dec = nn.Sequential(
            nn.Unflatten(1, (32, H, W)),
            nn.ConvTranspose2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, C, 3, padding=1), nn.Sigmoid()
        )
    def encoder(self, x):
        h = self.enc(x); z = self.fc_enc(h); return z
    def decoder(self, z):
        h = self.fc_dec(z); recon = self.dec(h); return recon
    def forward(self, x):
        z = self.encoder(x); recon = self.decoder(z); return z, recon

# -------------------------
# ResonantEngine (inference-time optimisation)
# -------------------------
class ResonantEngine:
    def __init__(self, model):
        self.model = model
        self.criterion = nn.MSELoss()
    def resonate(self, x_img, steps=None, lr=None):
        """
        x_img: (1,3,H,W) tensor on device
        returns: z_opt (1,latent), loss_val (float), z0_loss (float)
        """
        if steps is None: steps = CONFIG['inference_steps']
        if lr is None: lr = CONFIG['inference_lr']
        # initial latent (no grad)
        with torch.no_grad():
            z0 = self.model.encoder(x_img)        # [1,latent]
            recon0 = self.model.decoder(z0)
            z0_loss = float(self.criterion(recon0, x_img).item())
        # trainable latent
        z = z0.clone().detach().requires_grad_(True)
        opt = optim.Adam([z], lr=lr)
        loss_val = None
        for _ in range(steps):
            opt.zero_grad()
            recon = self.model.decoder(z)   # decoder uses model params (requires grad)
            loss = self.criterion(recon, x_img) # scalar -> has grad w.r.t. z
            loss.backward()
            opt.step()
            loss_val = float(loss.item())
        return z.detach(), loss_val, z0_loss

# -------------------------
# Cognitive Memory (gated)
# -------------------------
class CognitiveMemory:
    def __init__(self, capacity=2000, abs_thr=0.05, rel_factor=0.9):
        self.capacity = capacity
        self.abs_thr = abs_thr
        self.rel_factor = rel_factor
        self.vectors = deque(maxlen=capacity)
        self.coherences = deque(maxlen=capacity)
    def add_event(self, z_tensor, loss, z0_loss=None):
        store=False
        if loss < self.abs_thr:
            store=True
        elif (z0_loss is not None) and (loss < self.rel_factor * z0_loss):
            store=True
        if store:
            arr = z_tensor.squeeze(0).cpu().numpy().reshape(-1)
            self.vectors.append(arr)
            self.coherences.append(float(loss))
            return True
        return False
    def to_array(self):
        if len(self.vectors)==0:
            return np.zeros((0, CONFIG['latent_dim']))
        return np.vstack(list(self.vectors))
    def __len__(self):
        return len(self.vectors)

# -------------------------
# Symbol discovery (KMeans)
# -------------------------
def discover_symbols(memory_array, n_symbols=6):
    if memory_array.shape[0] == 0:
        return np.zeros((0, memory_array.shape[1])), np.array([])
    # If too few points, return unique points
    if memory_array.shape[0] < n_symbols:
        unique = np.unique(memory_array.round(decimals=8), axis=0)
        labels = np.zeros(memory_array.shape[0], dtype=int)
        for i, xi in enumerate(memory_array):
            labels[i] = int(np.argmin(np.sum((unique - xi)**2, axis=1)))
        return unique, labels
    km = KMeans(n_clusters=n_symbols, n_init=10, random_state=0)
    labels = km.fit_predict(memory_array)
    centers = km.cluster_centers_
    return centers, labels

# -------------------------
# ActorCritic (PPO)
# -------------------------
class ActorCritic(nn.Module):
    def __init__(self, input_dim, n_actions=4):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        self.policy = nn.Linear(128, n_actions)
        self.value = nn.Linear(128, 1)
    def forward(self, x):
        h = self.shared(x)
        logits = self.policy(h)
        value = self.value(h).squeeze(-1)
        return logits, value

# -------------------------
# Forward predictor for curiosity
# -------------------------
class ForwardModel(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
    def forward(self, x):
        return self.net(x)

# -------------------------
# GAE computation
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
# Training pipeline (main)
# -------------------------
def main():
    print("=== SRA Full RL Unified Script ===")
    env = SnakeEnv(size=CONFIG['grid_size'])

    # Build perceptual core
    ae = ConvAutoencoder(grid_size=CONFIG['grid_size'], latent_dim=CONFIG['latent_dim']).to(device)

    # ========== Phase 1: AE pretraining ==========
    print("[Phase 1] Collecting random states for AE pretraining...")
    samples = []
    while len(samples) < CONFIG['pretrain_samples']:
        obs = env.reset()
        for _ in range(30):
            a = random.randrange(4)
            obs, _, done, _ = env.step(a)
            samples.append(flatten_obs_for_ae(obs))
            if len(samples) >= CONFIG['pretrain_samples'] or done:
                break
    X = np.vstack(samples)  # shape (N, 3*H*W)
    N = X.shape[0]
    # build tensor for conv AE: (N,3,H,W)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device).view(-1, 3, CONFIG['grid_size'], CONFIG['grid_size'])

    print(f"[Phase 1] Training AE: samples={N}, epochs={CONFIG['ae_epochs']}, batch_size={CONFIG['ae_batch_size']}")
    opt_ae = optim.Adam(ae.parameters(), lr=CONFIG['ae_lr'])
    criterion = nn.MSELoss()
    ae.train()
    for ep in range(1, CONFIG['ae_epochs'] + 1):
        perm = torch.randperm(N)
        total = 0.0
        for i in range(0, N, CONFIG['ae_batch_size']):
            idx = perm[i:i+CONFIG['ae_batch_size']]
            batch = X_tensor[idx]
            opt_ae.zero_grad()
            z, recon = ae(batch)
            loss = criterion(recon, batch)
            loss.backward()
            opt_ae.step()
            total += float(loss.item()) * batch.size(0)
        avg = total / N
        if ep % 1 == 0:
            print(f" AE epoch {ep}/{CONFIG['ae_epochs']} - Loss {avg:.6f}")
    ae.eval()

    # ========== Resonant engine and memory ==========
    resonator = ResonantEngine(ae)
    memory = CognitiveMemory(capacity=CONFIG['memory_capacity'],
                             abs_thr=CONFIG['memory_abs_threshold'],
                             rel_factor=CONFIG['memory_rel_factor'])

    # ========== Phase 2: Exploration to populate memory ==========
    print("[Phase 2] Exploration to populate memory")
    for episode in range(1, CONFIG['explore_episodes'] + 1):
        obs = env.reset()
        ep_reward = 0.0
        steps = 0
        while True:
            # epsilon small exploration
            if random.random() < 0.08:
                a = random.randrange(4)
            else:
                # simple model-based lookahead: try actions, choose by resonant loss + shaping
                snap = env.copy_state()
                best_a = 0; best_score = -1e9
                dist_before = env.manhattan_head_to_food()
                for ac_try in range(4):
                    sim = SnakeEnv(env.size)
                    sim.set_state(snap)
                    sim_obs, r_sim, done_sim, _ = sim.step(ac_try)
                    x_img = obs_to_image_tensor(sim_obs)
                    z_opt, loss_opt, z0_loss = resonator.resonate(x_img)
                    score = -loss_opt
                    if r_sim > 0: score += 10.0
                    if done_sim: score -= 10.0
                    dist_after = sim.manhattan_head_to_food()
                    if dist_before is not None and dist_after is not None and dist_after < dist_before:
                        score += 0.5
                    if score > best_score:
                        best_score = score; best_a = ac_try
                a = best_a
            obs, reward, done, _ = env.step(a)
            steps += 1; ep_reward += reward

            # resonance on experienced obs -> memory gating
            x_img = obs_to_image_tensor(obs)
            z_opt, loss_opt, z0_loss = resonator.resonate(x_img)
            stored = memory.add_event(z_opt, loss_opt, z0_loss)
            if stored and CONFIG['debug']:
                print(f"  [mem+] ep={episode} step={steps} loss={loss_opt:.4f} z0={z0_loss:.4f} total_mem={len(memory)}")
            if steps > CONFIG['max_episode_steps']:
                break
            if done:
                break
        if CONFIG['debug']:
            print(f"[Explore {episode}] reward={ep_reward:.1f} mem_total={len(memory)}")
        if len(memory) >= CONFIG['min_memory_for_clustering']:
            print("[Phase 2] Enough memories collected, stop exploration early.")
            break

    # save memory
    mem_arr = memory.to_array()
    np.save(OUTDIR / "memory_vectors.npy", mem_arr)
    print(f"[Phase 2] Collected memories: {mem_arr.shape[0]}")

    if mem_arr.shape[0] < 2:
        print("Too few memories for clustering. Increase exploration or lower thresholds. Exiting.")
        return

    # ========== Phase 3: Symbol discovery ==========
    print("[Phase 3] Symbol discovery (KMeans)")
    centers, labels = discover_symbols(mem_arr, n_symbols=CONFIG['n_symbols'])
    print(f"[Phase 3] Discovered {centers.shape[0]} symbols.")
    np.save(OUTDIR / "symbol_centers.npy", centers)
    np.save(OUTDIR / "symbol_labels.npy", labels)

    # symbol embedding
    K = centers.shape[0]
    emb_dim = min(12, max(4, CONFIG['latent_dim']//2))
    symbol_embedding = nn.Embedding(K, emb_dim).to(device)
    # initialize embedding by mapping centers to vectors (mean-projection)
    # We keep embeddings trainable during PPO.

    # ========== Phase 4: PPO training ==========
    print("[Phase 4] PPO training with intrinsic curiosity")
    state_dim = CONFIG['latent_dim'] + emb_dim
    ac = ActorCritic(state_dim, n_actions=4).to(device)
    opt_ac = optim.Adam(ac.parameters(), lr=CONFIG['ppo_lr'])
    forward_model = ForwardModel(CONFIG['latent_dim'] + 4, CONFIG['latent_dim']).to(device)
    opt_fwd = optim.Adam(forward_model.parameters(), lr=CONFIG['forward_lr'])

    rollout_steps = CONFIG['rollout_steps']
    batch_size = rollout_steps
    for update in range(1, CONFIG['ppo_updates'] + 1):
        # rollout storage
        states = []
        actions = []
        logprobs = []
        rewards = []
        dones = []
        values = []
        intrinsics = []

        # collect rollout
        obs = env.reset()
        for t in range(rollout_steps):
            # get optimized latent & nearest symbol id
            x_img = obs_to_image_tensor(obs)
            z_opt, loss_opt, z0_loss = resonator.resonate(x_img)
            z_vec = z_opt.squeeze(0)  # tensor latent_dim
            z_np = z_vec.detach().cpu().numpy()
            # nearest symbol id
            dists = np.sum((centers - z_np)**2, axis=1)
            sym_id = int(np.argmin(dists))
            sym_idx = torch.tensor([sym_id], dtype=torch.long).to(device)
            sym_emb = symbol_embedding(sym_idx).squeeze(0)  # emb_dim

            state = torch.cat([z_vec, sym_emb], dim=0)  # state_dim
            state_tensor = state.unsqueeze(0)  # (1, state_dim)
            with torch.no_grad():
                logits, value = ac(state_tensor)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                action = int(dist.sample().item())
                logp = float(dist.log_prob(torch.tensor(action)).to(device))
                value = float(value.detach().cpu().item())

            # step env
            obs2, ext_reward, done, _ = env.step(action)
            # intrinsic curiosity: forward predictor
            # build predictor input: [z_vec (latent_dim) + action_onehot (4)]
            action_onehot = torch.zeros(4, device=device); action_onehot[action]=1.0
            fwd_in = torch.cat([z_vec.detach(), action_onehot], dim=0).unsqueeze(0)  # (1, latent+4)
            pred_z = forward_model(fwd_in)  # (1, latent)
            # get z_next by resonating obs2 (but do not optimize too many steps to keep speed)
            x2_img = obs_to_image_tensor(obs2)
            z2_opt, loss2_opt, z2_0 = resonator.resonate(x2_img)
            intrinsic = float(((pred_z.squeeze(0).detach().cpu().numpy() - z2_opt.detach().cpu().numpy().reshape(-1))**2).mean())
            total_reward = ext_reward + CONFIG['curiosity_beta'] * intrinsic

            # store rollout
            states.append(state.detach().cpu().numpy())
            actions.append(action)
            logprobs.append(logp)
            rewards.append(total_reward)
            dones.append(done)
            values.append(value)
            intrinsics.append(intrinsic)

            # update forward model online
            target = z2_opt.detach().squeeze(0)
            fwd_loss = F.mse_loss(pred_z.squeeze(0), target)
            opt_fwd.zero_grad(); fwd_loss.backward(); opt_fwd.step()

            obs = obs2
            if done:
                obs = env.reset()

        # convert rollout to numpy arrays
        states_np = np.vstack(states)
        actions_np = np.array(actions)
        old_logp_np = np.array(logprobs, dtype=np.float32)
        rewards_np = np.array(rewards, dtype=np.float32)
        dones_np = np.array(dones, dtype=np.float32)
        values_np = np.array(values, dtype=np.float32)

        # compute advantages & returns
        advs, returns = compute_gae(rewards_np, values_np, dones_np, CONFIG['gamma'], CONFIG['gae_lambda'])
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        # PPO updates
        N = states_np.shape[0]
        inds = np.arange(N)
        minibatch_size = max(1, N // CONFIG['ppo_minibatches'])
        for epoch in range(CONFIG['ppo_epochs']):
            np.random.shuffle(inds)
            for start in range(0, N, minibatch_size):
                mb = inds[start:start+minibatch_size]
                mb_states = torch.tensor(states_np[mb], dtype=torch.float32).to(device)
                mb_actions = torch.tensor(actions_np[mb], dtype=torch.long).to(device)
                mb_oldlog = torch.tensor(old_logp_np[mb], dtype=torch.float32).to(device)
                mb_adv = torch.tensor(advs[mb], dtype=torch.float32).to(device)
                mb_ret = torch.tensor(returns[mb], dtype=torch.float32).to(device)

                logits, vals = ac(mb_states)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                mb_logp = dist.log_prob(mb_actions)
                ratio = torch.exp(mb_logp - mb_oldlog)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - CONFIG['ppo_clip'], 1.0 + CONFIG['ppo_clip']) * mb_adv
                policy_loss = -torch.mean(torch.min(surr1, surr2))
                value_loss = F.mse_loss(vals, mb_ret)
                entropy = torch.mean(dist.entropy())
                loss = policy_loss + CONFIG['value_coef'] * value_loss - CONFIG['entropy_coef'] * entropy

                opt_ac.zero_grad(); loss.backward(); opt_ac.step()

        # logging per update
        mean_return = float(np.mean(rewards_np))
        mean_intr = float(np.mean(intrinsics))
        print(f"[PPO Update {update}/{CONFIG['ppo_updates']}] mean_return_rollout={mean_return:.3f} mean_intrinsic={mean_intr:.6f} mem={len(memory)}")

        # periodic saves
        if update % 10 == 0:
            torch.save(ac.state_dict(), OUTDIR / f"actorcrit_up{update}.pth")
            torch.save(forward_model.state_dict(), OUTDIR / f"forward_up{update}.pth")
            np.save(OUTDIR / "memory_vectors.npy", mem_arr)
            np.save(OUTDIR / "symbol_centers.npy", centers)

    # final save and simple plots
    torch.save(ac.state_dict(), OUTDIR / "actorcritic_final.pth")
    torch.save(forward_model.state_dict(), OUTDIR / "forward_final.pth")
    np.save(OUTDIR / "memory_vectors_final.npy", mem_arr)
    np.save(OUTDIR / "symbol_centers_final.npy", centers)

    print("Training finished. Artifacts saved to", OUTDIR)

# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    t0 = time.time()
    main()
    print("Elapsed:", time.time() - t0)

