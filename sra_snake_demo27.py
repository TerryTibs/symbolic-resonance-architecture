import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random

# --- CONFIG ---
CONFIG = {
    'grid_size': 10,
    'latent_dim': 64,
    'temporal_weight': 0.1,
    'num_episodes': 3,
    'max_steps': 50,
    'sigma3_horizon': 5
}

# --- ENVIRONMENT ---
class SnakeEnv:
    def __init__(self):
        self.grid_size = CONFIG['grid_size']
        self.action_space = 4  # up/down/left/right
        self.reset()

    def reset(self):
        self.snake = [(self.grid_size//2, self.grid_size//2)]
        self.food = [(random.randint(0,self.grid_size-1), random.randint(0,self.grid_size-1))]
        self.done = False
        return self._get_obs()

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True
        head_x, head_y = self.snake[0]
        if action==0: head_y-=1
        elif action==1: head_y+=1
        elif action==2: head_x-=1
        elif action==3: head_x+=1
        new_head = (max(0,min(self.grid_size-1,head_x)), max(0,min(self.grid_size-1,head_y)))
        reward = 0
        if new_head in self.snake:
            self.done=True
        else:
            self.snake.insert(0,new_head)
            if new_head in self.food:
                reward=1
                self.food=[(random.randint(0,self.grid_size-1), random.randint(0,self.grid_size-1))]
            else:
                self.snake.pop()
        return self._get_obs(), reward, self.done

    def _get_obs(self):
        obs = np.zeros((self.grid_size, self.grid_size))
        for x,y in self.snake:
            obs[y,x]=1
        for x,y in self.food:
            obs[y,x]=2
        return obs

    def simulate_step(self, action):
        # simple copy of step for internal prediction
        old_snake = self.snake.copy()
        old_food = self.food.copy()
        obs, reward, done = self.step(action)
        self.snake = old_snake
        self.food = old_food
        self.done=False
        return obs, reward

# --- CONV AUTOENCODER ---
class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.ReLU(),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*CONFIG['grid_size']*CONFIG['grid_size'], CONFIG['latent_dim'])
        )
        self.decoder = nn.Sequential(
            nn.Linear(CONFIG['latent_dim'], 32*CONFIG['grid_size']*CONFIG['grid_size']),
            nn.ReLU(),
            nn.Unflatten(1,(32, CONFIG['grid_size'], CONFIG['grid_size'])),
            nn.ConvTranspose2d(32,16,3,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16,1,3,padding=1),
            nn.Sigmoid()
        )
    def forward(self,x):
        z=self.encoder(x)
        return self.decoder(z), z

# --- RESONATOR ---
class Resonator:
    def __init__(self,ae):
        self.ae=ae
        self.memory_graph = nx.DiGraph()

    def resonate(self,x,z_prev=None,mem_sample=None):
        with torch.no_grad():
            z=self.ae.encoder(x)
        if z_prev is not None:
            z = 0.5*(z+z_prev)
        # optionally update memory graph
        if mem_sample is not None:
            self.memory_graph.add_node(mem_sample)
        return z

# --- VISUALIZATION ---
def render_overlay(env, memory_graph, sigma3_paths, ax):
    ax.clear()
    ax.set_xlim(-1,CONFIG['grid_size'])
    ax.set_ylim(-1,CONFIG['grid_size'])
    # snake body
    xs, ys = zip(*env.snake)
    ax.plot(xs, ys,'b-',label='snake')
    ax.scatter(xs, ys, c='b')
    # food
    for fx,fy in env.food:
        ax.scatter(fx,fy,c='g',s=100,label='food')
    # memory graph
    for n1,n2 in memory_graph.edges:
        xvals = [n1[0], n2[0]]
        yvals = [n1[1], n2[1]]
        ax.plot(xvals,yvals,'k--',alpha=0.3)
    # σ₃ predicted paths
    for path in sigma3_paths:
        if path:
            px,py=zip(*path)
            ax.plot(px,py,'r-',linewidth=2,alpha=0.7)
    ax.set_title("Live SRA v7 Snake Overlay")
    plt.pause(0.01)

# --- MAIN LOOP ---
def main():
    device='cpu'
    env=SnakeEnv()
    ae=ConvAE().to(device)
    resonator=Resonator(ae)
    fig,ax=plt.subplots(figsize=(6,6))
    plt.ion()

    beta = torch.tensor(1.0, requires_grad=True)
    memory_graph = resonator.memory_graph

    for episode in range(CONFIG['num_episodes']):
        obs = env.reset()
        obs_tensor = torch.tensor(obs,dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        z = None

        for step in range(CONFIG['max_steps']):
            mem_sample = random.choice(list(memory_graph.nodes)) if memory_graph.nodes else None
            z = resonator.resonate(obs_tensor,None if z is None else z, mem_sample)

            # Epistemic action selection
            action_scores=[]
            for a in range(env.action_space):
                obs_pred,_ = env.simulate_step(a)
                obs_tensor_pred = torch.tensor(obs_pred,dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                z_pred = resonator.resonate(obs_tensor_pred,z,mem_sample)
                recon = ae.decoder(z_pred)
                recon_loss = F.mse_loss(recon,obs_tensor_pred)
                temporal_loss = F.mse_loss(z_pred,z)
                epistemic_value = beta*(recon_loss+temporal_loss)
                action_scores.append(-epistemic_value.item())
            action = int(np.argmax(action_scores))

            obs,reward,done = env.step(action)
            obs_tensor = torch.tensor(obs,dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            # σ₃ paths (food-aware)
            sigma3_paths=[]
            hx,hy=env.snake[0]
            for f in env.food:
                path=[]
                for i in range(1,CONFIG['sigma3_horizon']+1):
                    nx_pos = min(hx+i,CONFIG['grid_size']-1)
                    ny_pos = hy
                    path.append((nx_pos,ny_pos))
                if f in path:
                    sigma3_paths.append(path)

            # update memory graph
            for idx in range(1,len(env.snake)):
                memory_graph.add_edge(env.snake[idx-1],env.snake[idx])

            # precision update
            beta = beta + 0.01*(reward-0.5)

            # render overlay
            render_overlay(env,memory_graph,sigma3_paths,ax)

            if done:
                print(f"Episode {episode+1} ended at step {step+1}")
                break
    plt.ioff()
    plt.show()

if __name__=="__main__":
    main()

