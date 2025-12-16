#!/usr/bin/env python3
"""
SRA v7: GRAND FUSION (Math + Scale)
===================================
The "Soul" of v2 (Resonance, Symbols, Curiosity)
+ The "Body" of v6 (DeepMind Eyes, Scent, Curriculum)
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.cluster import KMeans

# ==========================
# CONFIGURATION
# ==========================
CONFIG = {
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'output_dir': 'sra_v7_fusion',

    # World (Curriculum)
    'max_grid_size': 100,
    'start_grid_size': 10,
    'expand_step': 10,
    'max_episode_steps': 1000,

    # Perception (DeepMind Style)
    'latent_dim': 128,  # Compressed brain
    'ae_lr': 1e-3,

    # Resonance (The Math)
    'inference_steps': 2,    # Thinking steps per move (Active Inference)
    'inference_lr': 0.05,
    'temporal_weight': 0.5,
    'gravity_weight': 0.1,

    # Memory (Symbols)
    'memory_capacity': 5000,
    'n_symbols': 8,

    # PPO & Curiosity
    'ppo_updates': 1000,
    'rollout_size': 128,
    'ppo_lr': 3e-4,
    'curiosity_weight': 0.1, # Intrinsic Motivation
    'entropy_coef': 0.02,
}

# Setup
random.seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])
device = torch.device(CONFIG['device'])
os.makedirs(CONFIG['output_dir'], exist_ok=True)

print(f"--- SRA v7: GRAND FUSION ON {device} ---")

# ==========================
# ENVIRONMENT (Curriculum + Scent)
# ==========================
class CurriculumSnakeEnv:
    def __init__(self, max_size, current_size=10):
        self.max_size = max_size
        self.current_size = current_size
        self.offset = (self.max_size - self.current_size) // 2
        self.x_grid, self.y_grid = np.meshgrid(np.arange(max_size), np.arange(max_size))
        self.reset()

    def set_size(self, new_size):
        self.current_size = min(new_size, self.max_size)
        self.offset = (self.max_size - self.current_size) // 2
        self.reset()

    def reset(self):
        mid = self.offset + self.current_size // 2
        self.snake = [(mid, mid)]
        self.place_food()
        self.steps = 0
        self.done = False
        self.prev_dist = self._get_dist()
        return self._obs()

    def place_food(self):
        while True:
            fx = random.randint(self.offset, self.offset + self.current_size - 1)
            fy = random.randint(self.offset, self.offset + self.current_size - 1)
            if (fx, fy) not in self.snake:
                self.food = (fx, fy)
                break

    def _get_dist(self):
        if not self.food: return 0
        hx, hy = self.snake[0]
        fx, fy = self.food
        return abs(hx - fx) + abs(hy - fy)

    def step(self, action):
        if self.done: return self._obs(), 0.0, True, {}
        
        dx, dy = [(-1,0), (0,1), (1,0), (0,-1)][action]
        hx, hy = self.snake[0]
        nx, ny = hx + dx, hy + dy
        self.steps += 1

        if (nx < self.offset or nx >= self.offset + self.current_size or
            ny < self.offset or ny >= self.offset + self.current_size or
            (nx,ny) in self.snake):
            self.done = True
            return self._obs(), -1.0, True, {'outcome': 'die'}

        self.snake.insert(0, (nx,ny))
        
        reward = 0
        outcome = None

        if self.food and (nx,ny) == self.food:
            self.place_food()
            reward = 2.0 
            outcome = 'eat'
            self.prev_dist = self._get_dist()
        else:
            self.snake.pop()
            curr_dist = self._get_dist()
            reward = 0.01 if curr_dist < self.prev_dist else -0.01
            self.prev_dist = curr_dist
            
        if self.steps >= CONFIG['max_episode_steps']: self.done = True
        return self._obs(), reward, self.done, {'outcome': outcome}

    def _obs(self):
        obs = np.zeros((3, self.max_size, self.max_size), dtype=np.float32)
        hx, hy = self.snake[0]
        obs[0, hx, hy] = 1.0
        for s in self.snake[1:]: obs[1, s[0], s[1]] = 1.0
        if self.food:
            fx, fy = self.food
            dist_grid = np.abs(self.x_grid - fy) + np.abs(self.y_grid - fx)
            max_dist = self.max_size * 2
            scent = 1.0 - (dist_grid / max_dist)
            obs[2] = scent
        return obs

class VecCurriculumEnv:
    def __init__(self, num_envs, max_size, start_size):
        self.envs = [CurriculumSnakeEnv(max_size, start_size) for _ in range(num_envs)]
        self.current_size = start_size
    def expand(self, amount):
        self.current_size = min(self.current_size + amount, CONFIG['max_grid_size'])
        for env in self.envs: env.set_size(self.current_size)
        return self.current_size
    def reset(self): return np.stack([e.reset() for e in self.envs])
    def step(self, actions):
        results = [e.step(a) for e, a in zip(self.envs, actions)]
        obs, rews, dones, infos = zip(*results)
        new_obs = [self.envs[i].reset() if d else obs[i] for i, d in enumerate(dones)]
        return np.stack(new_obs), np.array(rews), np.array(dones), infos

# ==========================
# NETWORKS (Restored Autoencoder & Forward Model)
# ==========================
class DeepMindEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.flatten_dim = 64 * 9 * 9
        self.fc = nn.Linear(self.flatten_dim, CONFIG['latent_dim'])
    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = h.reshape(h.size(0), -1)
        return F.relu(self.fc(h))

class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DeepMindEncoder()
        # Decoder attempts to reconstruct the 100x100 grid (Simplified for speed)
        self.decoder = nn.Sequential(
            nn.Linear(CONFIG['latent_dim'], 64 * 9 * 9),
            nn.ReLU(),
            nn.Unflatten(1, (64, 9, 9)),
            nn.ConvTranspose2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, output_padding=0), nn.ReLU(), # Adj padding
            nn.ConvTranspose2d(32, 3, 8, stride=4, output_padding=0), nn.Sigmoid()
            # Note: Output size might need slight padding adjustment for 100x100 exactness
            # We use interpolation in the loss function to handle minor mismatches
        )
    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(CONFIG['latent_dim'], 128), nn.ReLU())
        self.actor = nn.Linear(128, 4)
        self.critic = nn.Linear(128, 1)
    def forward(self, z):
        h = self.shared(z)
        return self.actor(h), self.critic(h)

class ForwardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CONFIG['latent_dim'] + 4, 128), nn.ReLU(),
            nn.Linear(128, CONFIG['latent_dim'])
        )
    def forward(self, z, a):
        oh = F.one_hot(a, 4).float()
        return self.net(torch.cat([z, oh], 1))

# ==========================
# RESTORED: RESONANT ENGINE & MEMORY
# ==========================
class LivingKernel:
    def __init__(self):
        self.mem = []
    def add_batch(self, z_batch):
        # Sparse storage
        indices = np.random.choice(len(z_batch), size=max(1, len(z_batch)//10), replace=False)
        for i in indices:
            self.mem.append(z_batch[i])
            if len(self.mem) > CONFIG['memory_capacity']: self.mem.pop(0)
    def tensor(self):
        if not self.mem: return None
        sample = random.sample(self.mem, min(500, len(self.mem)))
        return torch.tensor(np.vstack(sample), device=device, dtype=torch.float32)

class ResonantEngine:
    def __init__(self, ae):
        self.ae = ae
    def resonate(self, x, z_prev, mem):
        # 1. Perception
        with torch.no_grad():
            z = self.ae.encoder(x)
        
        # 2. Optimization (The Math)
        z.requires_grad_(True)
        opt = optim.Adam([z], lr=CONFIG['inference_lr'])
        
        for _ in range(CONFIG['inference_steps']):
            opt.zero_grad()
            recon = self.ae.decoder(z)
            
            # Interpolate recon to match x size if decoder is slightly off
            if recon.shape != x.shape:
                recon = F.interpolate(recon, size=x.shape[2:])

            loss = F.mse_loss(recon, x)
            if z_prev is not None:
                loss += CONFIG['temporal_weight'] * F.mse_loss(z, z_prev)
            if mem is not None:
                d = torch.cdist(z, mem)
                min_dist, _ = torch.min(d, dim=1)
                loss += CONFIG['gravity_weight'] * torch.mean(min_dist)
            
            loss.backward()
            opt.step()
        
        return z.detach()

# ==========================
# TRAINING LOOP
# ==========================
def main():
    vec_env = VecCurriculumEnv(32, CONFIG['max_grid_size'], CONFIG['start_grid_size'])
    
    ae = ConvAE().to(device)
    ac = ActorCritic().to(device)
    fm = ForwardModel().to(device)
    
    opt_ae = optim.Adam(ae.parameters(), lr=CONFIG['ae_lr'])
    opt_ac = optim.Adam(ac.parameters(), lr=CONFIG['ppo_lr'])
    opt_fm = optim.Adam(fm.parameters(), lr=1e-3)
    
    kernel = LivingKernel()
    resonator = ResonantEngine(ae)
    
    # Pretrain AE briefly
    print("[Phase 1] Quick Eye Pretrain...")
    obs = vec_env.reset()
    for i in range(100):
        x = torch.tensor(obs, dtype=torch.float32).to(device)
        z, recon = ae(x)
        if recon.shape != x.shape: recon = F.interpolate(recon, size=x.shape[2:])
        loss = F.mse_loss(recon, x)
        opt_ae.zero_grad(); loss.backward(); opt_ae.step()
    
    print(f"[Phase 2] Fusion Training (Curriculum: {CONFIG['start_grid_size']}x{CONFIG['start_grid_size']})...")
    
    reward_history = []
    eats_in_window = 0
    z_prev = None
    
    for update in range(CONFIG['ppo_updates']):
        s_z, s_acts, s_logp, s_rews, s_vals, s_dones = [], [], [], [], [], []
        mem = kernel.tensor()

        # 1. Rollout with Resonance
        for step in range(CONFIG['rollout_size']):
            x = torch.tensor(obs, dtype=torch.float32).to(device)
            
            # ACTIVE INFERENCE
            z = resonator.resonate(x, z_prev, mem)
            
            # Action
            logits, val = ac(z)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            
            obs_new, r, d, infos = vec_env.step(a.cpu().numpy())
            
            # Intrinsic Curiosity
            z_pred = fm(z, a)
            with torch.no_grad(): z_target = ae.encoder(torch.tensor(obs_new, dtype=torch.float32).to(device))
            intrinsic = F.mse_loss(z_pred, z_target, reduction='none').mean(dim=1)
            
            total_r = torch.tensor(r, dtype=torch.float32).to(device) + CONFIG['curiosity_weight'] * intrinsic

            # Training Forward Model Online
            loss_fm = F.mse_loss(z_pred, z_target.detach())
            opt_fm.zero_grad(); loss_fm.backward(); opt_fm.step()
            
            # Store
            s_z.append(z)
            s_acts.append(a)
            s_logp.append(dist.log_prob(a))
            s_vals.append(val.squeeze())
            s_dones.append(torch.tensor(d, dtype=torch.float32).to(device))
            s_rews.append(total_r)
            
            # Curriculum stats
            for info in infos:
                if info.get('outcome') == 'eat': eats_in_window += 1
            
            obs = obs_new
            z_prev = z_pred.detach()
            # Mask state on done
            mask = (1.0 - torch.tensor(d, dtype=torch.float32).to(device)).unsqueeze(1)
            z_prev = z_prev * mask

        # Memory Update
        kernel.add_batch(z.cpu().numpy())

        # 2. PPO Update (Standard)
        with torch.no_grad():
            _, last_val = ac(z_prev)
            last_val = last_val.squeeze()
        
        returns, adv = [None]*CONFIG['rollout_size'], [None]*CONFIG['rollout_size']
        last_gae = 0
        for t in reversed(range(CONFIG['rollout_size'])):
            next_val = last_val if t == CONFIG['rollout_size']-1 else s_vals[t+1]
            next_non = 1.0 - s_dones[t]
            delta = s_rews[t] + 0.99 * next_val * next_non - s_vals[t]
            last_gae = delta + 0.99 * 0.95 * next_non * last_gae
            adv[t] = last_gae
            returns[t] = last_gae + s_vals[t]

        b_z = torch.stack(s_z).view(-1, CONFIG['latent_dim'])
        b_acts = torch.stack(s_acts).view(-1)
        b_logp = torch.stack(s_logp).view(-1).detach()
        b_adv = torch.stack(adv).view(-1).detach()
        b_ret = torch.stack(returns).view(-1).detach()
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        for _ in range(4):
            logits, v = ac(b_z)
            dist = torch.distributions.Categorical(logits=logits)
            loss_pol = -torch.min(torch.exp(dist.log_prob(b_acts)-b_logp)*b_adv, 
                                  torch.clamp(torch.exp(dist.log_prob(b_acts)-b_logp), 0.8, 1.2)*b_adv).mean()
            loss_val = 0.5 * F.mse_loss(v.squeeze(), b_ret)
            loss = loss_pol + loss_val - CONFIG['entropy_coef'] * dist.entropy().mean()
            opt_ac.zero_grad(); loss.backward(); opt_ac.step()

        # AE Training (Keep eyes sharp)
        # Using the batch of collected states
        # Re-construct image from Z
        batch_img = torch.stack([torch.tensor(vec_env.envs[0]._obs(), dtype=torch.float32).to(device) for _ in range(32)]) # Just grab current obs for speed
        z_out, recon_out = ae(batch_img)
        if recon_out.shape != batch_img.shape: recon_out = F.interpolate(recon_out, size=batch_img.shape[2:])
        loss_ae = F.mse_loss(recon_out, batch_img)
        opt_ae.zero_grad(); loss_ae.backward(); opt_ae.step()

        # Logs
        avg_r = torch.stack(s_rews).mean().item()
        reward_history.append(avg_r)

        if (update+1) % 10 == 0:
            print(f"  Update {update+1} | Grid: {vec_env.current_size}x{vec_env.current_size} | Avg Reward: {avg_r:.3f} | Eats: {eats_in_window}")
            
            # Curriculum Expansion
            if eats_in_window > 50 and vec_env.current_size < CONFIG['max_grid_size']:
                new_size = vec_env.expand(CONFIG['expand_step'])
                print(f"  >>> SYMBOLIC GROWTH: Expanding World to {new_size}x{new_size} <<<")
            eats_in_window = 0

    # Save
    torch.save({'ae_state': ae.state_dict(), 'ac_state': ac.state_dict(), 'config': CONFIG}, 
               os.path.join(CONFIG['output_dir'], 'sra_v7_brain.pth'))
    
    # Dreams
    if kernel.mem:
        print("Clustering Dreams (Symbol Extraction)...")
        mem_arr = np.vstack(kernel.mem)
        if len(mem_arr) > 1000: mem_arr = mem_arr[np.random.choice(len(mem_arr), 1000)]
        kmeans = KMeans(n_clusters=CONFIG['n_symbols'])
        centers = kmeans.fit_predict(mem_arr) # Just fit
        # Plotting dreams would require a perfect decoder which is tricky with interpolation,
        # but the symbols are stored in the weights now.

    print("DONE. Download 'sra_v7_brain.pth'.")

if __name__ == "__main__":
    main()
