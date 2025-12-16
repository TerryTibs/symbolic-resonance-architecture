#!/usr/bin/env python3
"""
sra_snake_ppo_tuned.py

SRA Snake with tuned PPO (Actor-Critic) + Curiosity + Symbolic Embeddings

- Autoencoder perceptual core + Resonant inference
- Coherence-gated memory -> adaptive clustering -> symbols
- Symbol embeddings + resonated latents = state representation
- PPO actor-critic training with GAE + curiosity intrinsic reward
- Training curves saved to disk

Author: Generated for user
Date: 2025-12-09
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
# CONFIG (tuned for efficiency)
# -------------------------
CONFIG = {
    'seed': 42,
    'device': 'cpu',  # set 'cuda' if available
    'deterministic': True,

    # environment
    'env_size': 8,
    'max_episode_steps': 300,

    # AE pretrain (small but enough)
    'collect_random_states': 1500,
    'random_max_steps': 40,
    'ae_epochs': 8,
    'ae_batch_size': 128,
    'ae_lr': 1e-3,

    # resonance (kept moderate)
    'latent_dim': 16,
    'resonance_steps': 8,
    'resonance_lr': 0.06,

    # memory gating
    'memory_abs_threshold': 0.012,
    'memory_rel_factor': 0.80,
    'memory_capacity': 2000,
    'min_memory_for_clustering': 60,

    # clustering
    'min_symbols': 3,
    'max_symbols': 10,

    # PPO hyperparameters (tuned)
    'ppo_epochs': 8,           # epochs per update
    'ppo_minibatches': 4,
    'ppo_clip': 0.2,
    'ppo_lr': 3e-4,
    'value_coef': 0.5,
    'entropy_coef': 0.01,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'batch_timesteps': 2048,   # timesteps collected per PPO update (smaller for sample efficiency)
    'max_updates': 200,        # number of PPO updates

    # curiosity (intrinsic) settings
    'curiosity_beta': 0.8,
    'forward_lr': 1e-3,

    # replay / artifacts
    'outdir': 'sra_snake_ppo_tuned_out',
    'save_every_updates': 10,
    'debug': True,
}

OUTDIR = Path(CONFIG['outdir'])
OUTDIR.mkdir(parents=True, exist_ok=True)

# deterministic seed
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if CONFIG.get('deterministic', False):
        torch.use_deterministic_algorithms(True)

set_seed(CONFIG['seed'])
device = torch.device(CONFIG['device'])

# -------------------------
# Simple Snake Env
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
# Perceptual Core (AE)
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
        z = self.encoder(x); return z, self.decoder(z)

# -------------------------
# Resonant engine (optimize z for x)
# -------------------------
class ResonantEngine:
    def __init__(self, model):
        self.model = model; self.criterion = nn.MSELoss()
    def resonate(self, x, steps=None, lr=None):
        if steps is None: steps = CONFIG['resonance_steps']
        if lr is None: lr = CONFIG['resonance_lr']
        with torch.no_grad():
            z0 = self.model.encoder(x); recon0 = self.model.decoder(z0); z0_loss = float(self.criterion(recon0, x).item())
        z = z0.clone().detach().requires_grad_(True)
        opt = optim.Adam([z], lr=lr)
        loss_val = None
        for _ in range(steps):
            opt.zero_grad()
            recon = self.model.decoder(z)
            loss = self.criterion(recon, x)
            loss.backward(); opt.step(); loss_val=float(loss.item())
        return z.detach(), loss_val, z0_loss

# -------------------------
# Memory (FIFO + gating)
# -------------------------
class CognitiveMemory:
    def __init__(self, capacity=2000, abs_thr=0.012, rel_factor=0.8):
        self.vectors = deque(maxlen=capacity); self.coherences = deque(maxlen=capacity)
        self.abs_thr = abs_thr; self.rel_factor = rel_factor
    def add_event(self, latent_vector, loss, z0_loss=None):
        store=False
        if loss < self.abs_thr: store=True
        elif z0_loss is not None and loss < self.rel_factor * z0_loss: store=True
        if store:
            self.vectors.append(latent_vector.detach().cpu().numpy().reshape(-1))
            self.coherences.append(loss); return True
        return False
    def to_array(self):
        if len(self.vectors)==0: return np.zeros((0,))
        return np.vstack(list(self.vectors))
    def __len__(self): return len(self.vectors)
    def clear(self): self.vectors.clear(); self.coherences.clear()

# -------------------------
# Adaptive clustering
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
        km=KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        inertias.append(km.inertia_); models.append(km)
    if len(inertias)==1: best_idx=0
    else:
        drops = np.diff(inertias); rel = drops/(np.array(inertias[:-1])+1e-8); best_idx=int(np.argmax(np.abs(rel)))+1
    best_km = models[best_idx]; return best_km.cluster_centers_, best_km.labels_, best_km.cluster_centers_.shape[0]

# -------------------------
# PPO networks: ActorCritic
# -------------------------
class ActorCritic(nn.Module):
    def __init__(self, input_dim, n_actions=4):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(input_dim,128), nn.ReLU(), nn.Linear(128,128), nn.ReLU())
        self.policy = nn.Sequential(nn.Linear(128, n_actions))
        self.value_head = nn.Linear(128,1)
    def forward(self, x):
        h = self.shared(x)
        logits = self.policy(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value

# -------------------------
# Forward model for curiosity (predict next latent from z + action)
# -------------------------
class ForwardModel(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim,128), nn.ReLU(), nn.Linear(128, latent_dim))
    def forward(self,x): return self.net(x)

# -------------------------
# Helpers: GAE, minibatching
# -------------------------
def compute_gae(rewards, values, dones, gamma, lam):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        if t == T-1:
            nextnonterminal = 1.0 - dones[t]
            nextvalue = values[t]  # bootstrap with last (we'll pass last value externally if needed)
        else:
            nextnonterminal = 1.0 - dones[t+1]
            nextvalue = values[t+1]
        delta = rewards[t] + gamma * nextvalue * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
    returns = advantages + values
    return advantages, returns

def flatten_batch(batch):
    return np.concatenate(batch, axis=0)

# -------------------------
# Save glyphs & PCA
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
            plt.savefig(sub/f"glyph_{i+1}.png"); plt.close()
    return sub

def save_memory_pca(memory, labels, tag):
    sub = OUTDIR / tag; sub.mkdir(parents=True, exist_ok=True)
    X = memory.to_array(); np.save(sub/"memory.npy", X); np.save(sub/"labels.npy", labels)
    try:
        pca = PCA(n_components=2); X2 = pca.fit_transform(X)
        plt.figure(figsize=(6,5)); plt.scatter(X2[:,0], X2[:,1], c=labels, s=8, cmap='tab10'); plt.title("Memory PCA")
        plt.savefig(sub/"memory_pca.png"); plt.close()
    except Exception as e:
        print("PCA failed:", e)

# -------------------------
# Main pipeline
# -------------------------
def main():
    env = SnakeEnv(size=CONFIG['env_size'])
    print("1) Collect random states for AE pretraining")
    data = []
    for _ in range(CONFIG['collect_random_states']):
        obs = env.reset()
        for _ in range(random.randint(1, CONFIG['random_max_steps'])):
            a = random.randrange(4); obs,_,done,_ = env.step(a); data.append(flatten_obs(obs))
            if done: break
    data = np.vstack(data); input_dim = data.shape[1]

    print("2) Build and train AE")
    core = PerceptualCore(input_dim=input_dim, latent_dim=CONFIG['latent_dim']).to(device)
    opt_ae = optim.Adam(core.parameters(), lr=CONFIG['ae_lr']); criterion = nn.MSELoss()
    N = data.shape[0]; tensor_data = torch.tensor(data, dtype=torch.float32).to(device)
    for ep in range(CONFIG['ae_epochs']):
        perm = torch.randperm(N)
        total=0.0
        core.train()
        for i in range(0, N, CONFIG['ae_batch_size']):
            idx = perm[i:i+CONFIG['ae_batch_size']]; batch = tensor_data[idx]
            opt_ae.zero_grad()
            z,recon = core(batch); loss = criterion(recon,batch)
            loss.backward(); opt_ae.step(); total += float(loss.item())*batch.size(0)
        print(f" AE epoch {ep+1}/{CONFIG['ae_epochs']} loss {total/N:.6f}")
    core.eval()

    engine = ResonantEngine(core)
    memory = CognitiveMemory(capacity=CONFIG['memory_capacity'],
                             abs_thr=CONFIG['memory_abs_threshold'],
                             rel_factor=CONFIG['memory_rel_factor'])

    print("3) Populate memory via SRA exploratory simulation")
    epsilon = 1.0
    for ep in range(1, 25+1):   # quick exploration phase (25 episodes)
        obs = env.reset(); ep_reward=0.0
        for t in range(CONFIG['max_episode_steps']):
            # epsilon-greedy SRA (simulate each action)
            if random.random() < epsilon:
                a = random.randrange(4)
            else:
                # simulate each action and pick best score (low recon loss + shaping favorable)
                base = env.copy_state(); best_score=-1e9; best_a=0
                dist_before = env.manhattan_head_to_food()
                for a_try in range(4):
                    sim = SnakeEnv(env.size); sim.set_state(base); obs_sim, r_sim, done_sim, _ = sim.step(a_try)
                    dist_after = sim.manhattan_head_to_food()
                    shaping = 0.5 * (dist_before - dist_after) if (dist_before is not None and dist_after is not None) else 0.0
                    x = obs_to_tensor(obs_sim)
                    z_opt, loss_opt, z0_loss = engine.resonate(x)
                    score = -loss_opt + shaping
                    if r_sim > 0: score += 10.0
                    if done_sim: score -= 10.0
                    if score > best_score:
                        best_score=score; best_a=a_try
                a = best_a
            obs, reward, done, _ = env.step(a); ep_reward += reward
            x = obs_to_tensor(obs); z_opt, loss_opt, z0_loss = engine.resonate(x)
            memory.add_event(z_opt, loss_opt, z0_loss)
            if done: break
        epsilon = max(0.02, epsilon * 0.9)
        if CONFIG['debug']: print(f"[explore ep {ep}] reward={ep_reward:.1f} mem={len(memory)} eps={epsilon:.3f}")

    if len(memory) < CONFIG['min_memory_for_clustering']:
        print("Not enough memory collected; increase exploration.")
        return

    centers, labels, k = adaptive_symbol_discovery(memory.vectors, min_k=CONFIG['min_symbols'], max_k=CONFIG['max_symbols'])
    print(f"Symbol discovery -> k={k}")
    save_glyphs(core, centers, tag="symbols_initial"); save_memory_pca(memory, labels, tag="mem_initial")

    # symbol embedding
    K = centers.shape[0]
    symbol_emb_dim = 8
    symbol_embedding = nn.Embedding(K, symbol_emb_dim).to(device)

    # PPO agent
    state_dim = CONFIG['latent_dim'] + symbol_emb_dim
    ac = ActorCritic(state_dim, n_actions=4).to(device)
    optimizer = optim.Adam(ac.parameters(), lr=CONFIG['ppo_lr'])
    forward_model = ForwardModel(CONFIG['latent_dim'] + 4, CONFIG['latent_dim']).to(device)
    fwd_opt = optim.Adam(forward_model.parameters(), lr=CONFIG['forward_lr'])

    # training buffers for PPO
    ep_returns = []
    ep_intrinsics = []
    policy_losses = []
    value_losses = []
    entropies = []

    # helper to get state vector from obs
    def get_state(obs):
        with torch.no_grad():
            x = obs_to_tensor(obs)
            z_opt, loss_opt, z0_loss = engine.resonate(x)
            z_np = z_opt.detach().cpu().numpy().reshape(-1)
            # symbol id by nearest center
            diffs = ((centers - z_np)**2).sum(axis=1)
            sym_id = int(np.argmin(diffs))
            sym_tensor = torch.tensor([sym_id], dtype=torch.long).to(device)
            sym_emb = symbol_embedding(sym_tensor).squeeze(0)
            state = torch.cat([z_opt.squeeze(0), sym_emb], dim=0).unsqueeze(0)  # 1 x D
            return state, z_opt.squeeze(0).detach(), sym_id

    # PPO main loop: collect batch_timesteps then update
    total_steps = 0
    for update in range(1, CONFIG['max_updates']+1):
        # collect rollout
        obs = env.reset()
        roll_states = []; roll_actions = []; roll_logprobs = []; roll_rewards = []; roll_values = []; roll_dones = []; roll_z = []; roll_intrinsic = []
        batch_steps = 0
        while batch_steps < CONFIG['batch_timesteps']:
            state, z_vec, sym_id = get_state(obs)
            logits, value = ac(state)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = int(dist.sample().item())
            logp = float(dist.log_prob(torch.tensor(action)).item())
            # step
            obs2, ext_reward, done, _ = env.step(action)
            # compute intrinsic reward via forward model error
            z2_opt, loss2_opt, z2_0 = engine.resonate(obs_to_tensor(obs2))
            a_onehot = torch.zeros(4, device=device); a_onehot[action]=1.0
            fwd_in = torch.cat([z_vec.to(device), a_onehot], dim=0).unsqueeze(0)
            pred = forward_model(fwd_in)
            intrinsic = float(((pred.squeeze(0).detach().cpu().numpy() - z2_opt.detach().cpu().numpy().reshape(-1))**2).mean())
            total_reward = ext_reward + CONFIG['curiosity_beta'] * intrinsic
            # store rollout step
            roll_states.append(state.squeeze(0).detach().cpu().numpy())
            roll_actions.append(action)
            roll_logprobs.append(logp)
            roll_rewards.append(total_reward)
            roll_values.append(float(value.detach().cpu().numpy()))
            roll_dones.append(done)
            roll_z.append(z_vec.detach().cpu().numpy())
            roll_intrinsic.append(intrinsic)
            # train forward model on-the-fly (supervised)
            pred_tensor = pred.squeeze(0)
            target_z2 = z2_opt.detach()
            fwd_loss = ((pred_tensor - target_z2)**2).mean()
            fwd_opt.zero_grad(); fwd_loss.backward(); fwd_opt.step()
            obs = obs2
            batch_steps += 1; total_steps += 1
            if done:
                obs = env.reset()
        # convert rollout to numpy arrays
        states_np = np.vstack(roll_states)
        actions_np = np.array(roll_actions)
        old_logprobs_np = np.array(roll_logprobs)
        rewards_np = np.array(roll_rewards, dtype=np.float32)
        values_np = np.array(roll_values, dtype=np.float32)
        dones_np = np.array(roll_dones, dtype=np.float32)
        intrinsics_np = np.array(roll_intrinsic, dtype=np.float32)

        # compute advantages and returns using GAE
        advantages, returns = compute_gae(rewards_np, values_np, dones_np, CONFIG['gamma'], CONFIG['gae_lambda'])
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update: multiple epochs, minibatches
        N = states_np.shape[0]
        inds = np.arange(N)
        minibatch_size = max(1, N // CONFIG['ppo_minibatches'])
        for epoch in range(CONFIG['ppo_epochs']):
            np.random.shuffle(inds)
            for start in range(0, N, minibatch_size):
                mbinds = inds[start:start+minibatch_size]
                mb_states = torch.tensor(states_np[mbinds], dtype=torch.float32).to(device)
                mb_actions = torch.tensor(actions_np[mbinds], dtype=torch.long).to(device)
                mb_oldlog = torch.tensor(old_logprobs_np[mbinds], dtype=torch.float32).to(device)
                mb_adv = torch.tensor(advantages[mbinds], dtype=torch.float32).to(device)
                mb_ret = torch.tensor(returns[mbinds], dtype=torch.float32).to(device)

                logits, vals = ac(mb_states)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                mb_logp = dist.log_prob(mb_actions)
                ratio = torch.exp(mb_logp - mb_oldlog)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - CONFIG['ppo_clip'], 1.0 + CONFIG['ppo_clip']) * mb_adv
                policy_loss = -torch.mean(torch.min(surr1, surr2))
                value_loss = torch.mean((vals - mb_ret)**2)
                entropy = torch.mean(dist.entropy())

                loss = policy_loss + CONFIG['value_coef'] * value_loss - CONFIG['entropy_coef'] * entropy
                optimizer.zero_grad(); loss.backward(); optimizer.step()

        # logging: compute average metrics
        avg_return = np.sum(rewards_np) / (np.sum(dones_np==1) + 1e-8) if np.sum(dones_np==1)>0 else np.mean(rewards_np)
        ep_returns.append(np.sum(rewards_np))
        ep_intrinsics.append(np.sum(intrinsics_np))
        policy_losses.append(float(policy_loss.detach().cpu().numpy()))
        value_losses.append(float(value_loss.detach().cpu().numpy()))
        entropies.append(float(entropy.detach().cpu().numpy()))
        if (update % CONFIG['save_every_updates']) == 0:
            torch.save(ac.state_dict(), OUTDIR/f"actorcritic_up{update}.pth")
            torch.save(forward_model.state_dict(), OUTDIR/f"forward_up{update}.pth")
            np.save(OUTDIR/"memory.npy", memory.to_array())
            print(f"[Update {update}] saved checkpoints. avg_return_rollout={np.mean(ep_returns[-10:]):.3f}")

        if CONFIG['debug']:
            print(f"[Update {update}] steps={total_steps} mean_return_last={np.mean(ep_returns[-5:]):.3f} mean_intrinsic_last={np.mean(ep_intrinsics[-5:]):.6f}")

        # optional: recluster every few updates (disabled for simplicity)
        # if update % 50 == 0:
        #     centers, labels, k = adaptive_symbol_discovery(memory.vectors, min_k=CONFIG['min_symbols'], max_k=CONFIG['max_symbols'])

        # stop condition (if desired)
        if update >= CONFIG['max_updates']:
            break

    # final saves and plots
    torch.save(ac.state_dict(), OUTDIR/"actorcritic_final.pth")
    torch.save(forward_model.state_dict(), OUTDIR/"forward_final.pth")
    np.save(OUTDIR/"memory_final.npy", memory.to_array())
    save_glyphs(core, centers, tag="symbols_final")
    save_memory_pca(memory, labels, tag="mem_final")

    # Plot training curves
    plt.figure(figsize=(8,4)); plt.plot(ep_returns, label='rollout_return'); plt.xlabel('update'); plt.ylabel('sum return'); plt.legend(); plt.tight_layout(); plt.savefig(OUTDIR/"return_curve.png"); plt.close()
    plt.figure(figsize=(8,4)); plt.plot(ep_intrinsics, label='intrinsic_sum'); plt.xlabel('update'); plt.ylabel('intrinsic'); plt.legend(); plt.tight_layout(); plt.savefig(OUTDIR/"intrinsic_curve.png"); plt.close()
    plt.figure(figsize=(8,4)); plt.plot(policy_losses, label='policy_loss'); plt.plot(value_losses, label='value_loss'); plt.legend(); plt.tight_layout(); plt.savefig(OUTDIR/"losses.png"); plt.close()
    print("Training finished. Artifacts in:", OUTDIR)

if __name__ == "__main__":
    t0 = time.time(); main(); print("Elapsed:", time.time() - t0)

