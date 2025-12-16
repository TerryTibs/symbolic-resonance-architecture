#!/usr/bin/env python3
"""
SRA v7: GRAND FUSION (Full Epistemic Active Inference + Multi-level σ3 Visualization)
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
import networkx as nx

# ==========================
# CONFIGURATION
# ==========================
CONFIG = {
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'output_dir': 'sra_v7_full',
    
    # World
    'max_grid_size': 50,
    'start_grid_size': 10,
    'expand_step': 5,
    'max_episode_steps': 200,
    
    # Perception
    'latent_dim': 64,
    'ae_lr': 1e-3,
    
    # Resonance
    'inference_steps': 3,
    'inference_lr': 0.05,
    'temporal_weight': 0.5,
    'gravity_weight': 0.1,
    
    # Memory
    'memory_capacity': 2000,
    'n_symbols': 8,
    
    # Active Inference
    'planning_horizon': 5,
    'epistemic_weight': 0.5,
}

# Setup
random.seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])
device = torch.device(CONFIG['device'])
os.makedirs(CONFIG['output_dir'], exist_ok=True)

# ==========================
# ENVIRONMENT
# ==========================
class CurriculumSnakeEnv:
    def __init__(self, max_size, current_size=10):
        self.max_size = max_size
        self.current_size = current_size
        self.offset = (max_size - current_size)//2
        self.x_grid, self.y_grid = np.meshgrid(np.arange(max_size), np.arange(max_size))
        self.reset()
    
    def set_size(self, new_size):
        self.current_size = min(new_size, self.max_size)
        self.offset = (self.max_size - self.current_size)//2
        self.reset()
    
    def reset(self):
        mid = self.offset + self.current_size//2
        self.snake = [(mid, mid)]
        self.place_food()
        self.steps = 0
        self.done = False
        self.prev_dist = self._get_dist()
        return self._obs()
    
    def place_food(self):
        while True:
            fx = random.randint(self.offset, self.offset+self.current_size-1)
            fy = random.randint(self.offset, self.offset+self.current_size-1)
            if (fx, fy) not in self.snake:
                self.food = (fx, fy)
                break
    
    def _get_dist(self):
        if not self.food: return 0
        hx, hy = self.snake[0]
        fx, fy = self.food
        return abs(hx-fx)+abs(hy-fy)
    
    def step(self, action):
        if self.done: return self._obs(), 0.0, True, {}
        dx, dy = [(-1,0),(0,1),(1,0),(0,-1)][action]
        hx, hy = self.snake[0]
        nx, ny = hx+dx, hy+dy
        self.steps +=1
        if nx<self.offset or nx>=self.offset+self.current_size or ny<self.offset or ny>=self.offset+self.current_size or (nx,ny) in self.snake:
            self.done=True
            return self._obs(), -1.0, True, {'outcome':'die'}
        self.snake.insert(0,(nx,ny))
        reward = 0.0
        outcome = None
        if (nx,ny)==self.food:
            self.place_food()
            reward=2.0
            outcome='eat'
            self.prev_dist=self._get_dist()
        else:
            self.snake.pop()
            curr_dist=self._get_dist()
            reward=0.01 if curr_dist<self.prev_dist else -0.01
            self.prev_dist=curr_dist
        if self.steps>=CONFIG['max_episode_steps']: self.done=True
        return self._obs(), reward, self.done, {'outcome':outcome}
    
    def _obs(self):
        obs=np.zeros((3,self.max_size,self.max_size),dtype=np.float32)
        hx,hy=self.snake[0]
        obs[0,hx,hy]=1.0
        for s in self.snake[1:]: obs[1,s[0],s[1]]=1.0
        if self.food:
            fx,fy=self.food
            dist_grid=np.abs(self.x_grid-fy)+np.abs(self.y_grid-fx)
            max_dist=self.max_size*2
            scent=1.0-(dist_grid/max_dist)
            obs[2]=scent
        return obs

class VecCurriculumEnv:
    def __init__(self,num_envs,max_size,start_size):
        self.envs=[CurriculumSnakeEnv(max_size,start_size) for _ in range(num_envs)]
        self.current_size=start_size
    def expand(self,amount):
        self.current_size=min(self.current_size+amount, CONFIG['max_grid_size'])
        for env in self.envs: env.set_size(self.current_size)
        return self.current_size
    def reset(self): return np.stack([e.reset() for e in self.envs])
    def step(self,actions):
        results=[e.step(a) for e,a in zip(self.envs,actions)]
        obs, rews, dones, infos=zip(*results)
        new_obs=[self.envs[i].reset() if d else obs[i] for i,d in enumerate(dones)]
        return np.stack(new_obs), np.array(rews), np.array(dones), infos

# ==========================
# AUTOENCODER
# ==========================
class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Conv2d(3,32,3,stride=2,padding=1), nn.ReLU(),
            nn.Conv2d(32,64,3,stride=2,padding=1), nn.ReLU(),
            nn.Conv2d(64,64,3,stride=2,padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7,CONFIG['latent_dim']), nn.ReLU()
        )
        self.decoder=nn.Sequential(
            nn.Linear(CONFIG['latent_dim'],64*7*7), nn.ReLU(),
            nn.Unflatten(1,(64,7,7)),
            nn.ConvTranspose2d(64,64,3,stride=2,padding=1,output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64,32,3,stride=2,padding=1,output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32,3,3,stride=2,padding=1,output_padding=1), nn.Sigmoid()
        )
    def forward(self,x):
        z=self.encoder(x)
        recon=self.decoder(z)
        return z,recon

# ==========================
# RESONANT ENGINE
# ==========================
class LivingKernel:
    def __init__(self):
        self.mem=[]
    def add_batch(self,z_batch):
        indices=np.random.choice(len(z_batch), size=max(1,len(z_batch)//10), replace=False)
        for i in indices:
            self.mem.append(z_batch[i])
            if len(self.mem)>CONFIG['memory_capacity']: self.mem.pop(0)
    def tensor(self):
        if not self.mem: return None
        sample=random.sample(self.mem,min(500,len(self.mem)))
        return torch.tensor(np.vstack(sample),device=device,dtype=torch.float32)

class ResonantEngine:
    def __init__(self,ae):
        self.ae=ae
    def resonate(self,x,z_prev,mem):
        with torch.no_grad(): z0=self.ae.encoder(x)
        z=z0.clone().detach().requires_grad_(True)
        opt=optim.Adam([z],lr=CONFIG['inference_lr'])
        for _ in range(CONFIG['inference_steps']):
            opt.zero_grad()
            recon=self.ae.decoder(z)
            if recon.shape!=x.shape:
                recon=F.interpolate(recon,size=x.shape[2:])
            loss=F.mse_loss(recon,x)
            if z_prev is not None:
                loss+=CONFIG['temporal_weight']*F.mse_loss(z,z_prev)
            if mem is not None:
                d=torch.cdist(z,mem)
                min_dist,_=torch.min(d,dim=1)
                loss+=CONFIG['gravity_weight']*torch.mean(min_dist)
            loss.backward()
            opt.step()
        return z.detach()

# ==========================
# ACTIVE INFERENCE POLICY
# ==========================
def epistemic_action_selection(env,z,ae,kernel):
    best_a=None
    min_fe=float('inf')
    mem=kernel.tensor()
    for a in range(4):
        hx,hy=env.snake[0]
        dx,dy=[(-1,0),(0,1),(1,0),(0,-1)][a]
        nx,ny=hx+dx,hy+dy
        if nx<env.offset or nx>=env.offset+env.current_size or ny<env.offset or ny>=env.offset+env.current_size or (nx,ny) in env.snake:
            continue
        # simulate next observation
        sim_snake=[(nx,ny)]+env.snake[:-1]
        sim_obs=np.zeros_like(env._obs())
        sim_obs[0,nx,ny]=1.0
        for s in sim_snake[1:]: sim_obs[1,s[0],s[1]]=1.0
        if env.food: fx,fy=env.food; dist_grid=np.abs(env.x_grid-fy)+np.abs(env.y_grid-fx); sim_obs[2]=1.0-(dist_grid/(env.max_size*2))
        sim_obs_tensor=torch.tensor(sim_obs[None],dtype=torch.float32,device=device)
        z_sim,_=ae(sim_obs_tensor)
        fe_loss=F.mse_loss(z,z_sim)
        if mem is not None:
            fe_loss+=CONFIG['gravity_weight']*torch.mean(torch.min(torch.cdist(z_sim,mem),dim=1)[0])
        if fe_loss.item()<min_fe:
            min_fe=fe_loss.item(); best_a=a
    if best_a is None: best_a=random.randint(0,3)
    return best_a

# ==========================
# VISUALIZATION
# ==========================
def visualize(env,kernel,sigma3_paths):
    plt.clf()
    G=nx.Graph()
    mem_nodes=[]
    if kernel.mem:
        for i,vec in enumerate(kernel.mem):
            G.add_node(i,pos=vec[:2])
            mem_nodes.append(i)
    pos={i:(vec[0],vec[1]) for i,vec in enumerate(kernel.mem)}
    nx.draw(G,pos,node_color='blue',alpha=0.3,with_labels=False)
    # σ3 paths
    for path in sigma3_paths:
        xs=[p[0] for p in path]; ys=[p[1] for p in path]
        plt.plot(xs,ys,'r--')
    # snake
    snake_x=[s[0] for s in env.snake]; snake_y=[s[1] for s in env.snake]
    plt.plot(snake_x,snake_y,'g-',linewidth=2)
    # food
    fx,fy=env.food; plt.plot(fx,fy,'yo',markersize=8)
    plt.xlim(0,env.max_size); plt.ylim(0,env.max_size)
    plt.pause(0.001)

# ==========================
# TRAINING LOOP
# ==========================
def main():
    vec_env=VecCurriculumEnv(1,CONFIG['max_grid_size'],CONFIG['start_grid_size'])
    ae=ConvAE().to(device)
    kernel=LivingKernel()
    resonator=ResonantEngine(ae)
    
    obs=vec_env.reset()[0]
    plt.ion(); fig=plt.figure(figsize=(6,6))
    
    for step in range(200):
        obs_tensor=torch.tensor(obs[None],dtype=torch.float32,device=device)
        mem_sample=kernel.tensor()
        z=resonator.resonate(obs_tensor,None if step==0 else z,mem_sample)
        
        # Active Inference action
        action=epistemic_action_selection(vec_env.envs[0],z,ae,kernel)
        obs,reward,done,info=vec_env.step([action])
        obs=obs[0]
        kernel.add_batch(z.cpu().numpy())
        
        # σ3 multi-step paths (future head positions)
        sigma3_paths=[]
        hx,hy=vec_env.envs[0].snake[0]
        path=[(hx,hy)]
        for h in range(1,CONFIG['planning_horizon']):
            hx2,hy2=path[-1]; best_a=random.randint(0,3)
            dx,dy=[(-1,0),(0,1),(1,0),(0,-1)][best_a]; hx2+=dx; hy2+=dy
            path.append((hx2,hy2))
        sigma3_paths.append(path)
        
        visualize(vec_env.envs[0],kernel,sigma3_paths)
        if done: obs=vec_env.reset()[0]
    
    plt.ioff(); plt.show()

if __name__=="__main__":
    main()

