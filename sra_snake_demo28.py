import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

# -------------------------
# CONFIG
# -------------------------
CONFIG = {
    'grid_size': 30,
    'latent_dim': 64,
    'temporal_weight': 0.1,
    'max_steps': 1000,
    'num_games': 10
}

device = torch.device("cpu")

# -------------------------
# SNAKE ENVIRONMENT
# -------------------------
class SnakeEnv:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.snake = deque([[self.grid_size//2, self.grid_size//2]])
        self.direction = np.array([0,1])
        self.food = self._place_food()
        self.done = False
        self.steps = 0
        return self._get_obs()

    def _place_food(self):
        while True:
            pos = np.random.randint(0,self.grid_size,2).tolist()
            if pos not in self.snake:
                return pos

    def step(self, action):
        # action: 0=up,1=down,2=left,3=right
        if action==0: self.direction=[-1,0]
        elif action==1: self.direction=[1,0]
        elif action==2: self.direction=[0,-1]
        elif action==3: self.direction=[0,1]

        new_head = (np.array(self.snake[0])+self.direction).tolist()
        self.snake.appendleft(new_head)

        reward = 0
        self.done=False
        if new_head==self.food:
            reward=1
            self.food=self._place_food()
        else:
            self.snake.pop()

        if (new_head in list(self.snake)[1:] or 
            not all(0<=c<self.grid_size for c in new_head)):
            self.done=True
            reward=-1

        self.steps+=1
        if self.steps>=CONFIG['max_steps']:
            self.done=True

        return self._get_obs(), reward, self.done

    def _get_obs(self):
        grid=np.zeros((self.grid_size,self.grid_size))
        for x,y in self.snake:
            grid[x,y]=1
        fx,fy=self.food
        grid[fx,fy]=2
        return grid

# -------------------------
# CONV AUTOENCODER
# -------------------------
class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,16,3,padding=1)
        self.conv2=nn.Conv2d(16,32,3,padding=1)
        self.conv3=nn.Conv2d(32,64,3,padding=1)
        # compute flattened size
        dummy=torch.zeros(1,1,CONFIG['grid_size'],CONFIG['grid_size'])
        h=self._encode(dummy)
        self.fc=nn.Linear(h.numel(),CONFIG['latent_dim'])
        self.decoder_fc=nn.Linear(CONFIG['latent_dim'],h.numel())
        self.convT3=nn.ConvTranspose2d(64,32,3,padding=1)
        self.convT2=nn.ConvTranspose2d(32,16,3,padding=1)
        self.convT1=nn.ConvTranspose2d(16,1,3,padding=1)

    def _encode(self,x):
        h=F.relu(self.conv1(x))
        h=F.relu(self.conv2(h))
        h=F.relu(self.conv3(h))
        return h

    def encode(self,x):
        h=self._encode(x)
        h_flat=h.view(h.size(0),-1)
        z=self.fc(h_flat)
        return z

    def decode(self,z):
        h=self.decoder_fc(z)
        h=h.view(z.size(0),64,CONFIG['grid_size'],CONFIG['grid_size'])
        h=F.relu(self.convT3(h))
        h=F.relu(self.convT2(h))
        h=torch.sigmoid(self.convT1(h))
        return h

# -------------------------
# RESONATOR / MEMORY
# -------------------------
class Resonator:
    def __init__(self,ae):
        self.ae=ae
        self.memory=[]

    def resonate(self,x,z_prev,mem_sample):
        with torch.no_grad():
            z0=self.ae.encode(x)
        if z_prev is None:
            z=z0
        else:
            z=z0*0.5+z_prev*0.5
        return z

# -------------------------
# SELECT ACTION
# -------------------------
def select_action(env,resonator,z):
    scores=[]
    for a in range(4):
        obs,_ ,done= env.step(a)
        obs_tensor=torch.tensor(obs,dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        z_pred=resonator.ae.encode(obs_tensor)
        scores.append(-torch.var(z_pred).detach().cpu().numpy())
        # revert step
        env.snake.popleft()
        if len(env.snake)<1:
            env.reset()
    return int(np.argmax(scores))

# -------------------------
# VISUALIZATION
# -------------------------
def plot_state(env,G=None,paths=None):
    plt.clf()
    grid=env._get_obs()
    plt.imshow(grid,cmap='gray_r',vmin=0,vmax=2)
    if G:
        pos={n:(n[1],env.grid_size-1-n[0]) for n in G.nodes()}
        nx.draw(G,pos,node_size=50,node_color='blue',alpha=0.5)
    if paths:
        for path in paths:
            xs=[p[1] for p in path]
            ys=[env.grid_size-1-p[0] for p in path]
            plt.plot(xs,ys,'r-')
    plt.pause(0.05)

# -------------------------
# MAIN
# -------------------------
def run_game():
    env=SnakeEnv(CONFIG['grid_size'])
    ae=ConvAE().to(device)
    resonator=Resonator(ae)

    plt.ion()
    fig=plt.figure()

    for game in range(CONFIG['num_games']):
        obs=env.reset()
        z=None
        G=nx.DiGraph()
        for step in range(CONFIG['max_steps']):
            obs_tensor=torch.tensor(obs,dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            z=resonator.resonate(obs_tensor,None if step==0 else z,None)

            # simple σ3 path foresight
            paths=[]
            head=env.snake[0]
            for i in range(3):
                next_pos=(head[0]+i*env.direction[0],head[1]+i*env.direction[1])
                paths.append([head,next_pos])

            a=select_action(env,resonator,z)
            obs,reward,done=env.step(a)
            # memory graph update
            G.add_node(tuple(env.snake[0]))
            if len(env.snake)>1:
                G.add_edge(tuple(env.snake[1]),tuple(env.snake[0]))

            plot_state(env,G,paths)
            if done:
                break
    plt.ioff()
    plt.show()

# -------------------------
# RUN
# -------------------------
if __name__=="__main__":
    run_game()

