import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random

# =======================
# CONFIG
# =======================
CONFIG = {
    'grid_size': 20,
    'latent_dim': 64,
    'temporal_weight': 0.1,
    'epistemic_weight': 0.5,
    'rollout_steps': 500
}

# =======================
# ENVIRONMENT
# =======================
class SnakeEnv:
    def __init__(self, size=CONFIG['grid_size']):
        self.current_size = size
        self.reset()

    def reset(self):
        self.snake = [(self.current_size//2, self.current_size//2)]
        self.direction = (0,1)
        self.food = (random.randint(0,self.current_size-1), random.randint(0,self.current_size-1))
        self.done = False
        return self.get_obs()

    def get_obs(self):
        obs = np.zeros((self.current_size, self.current_size), dtype=np.float32)
        for x,y in self.snake:
            obs[x,y]=1
        obs[self.food[0], self.food[1]] = 2
        return obs

    def step(self, action):
        if action == 0: self.direction = (0,1)
        elif action == 1: self.direction = (0,-1)
        elif action == 2: self.direction = (-1,0)
        elif action == 3: self.direction = (1,0)
        new_head = (self.snake[0][0]+self.direction[0], self.snake[0][1]+self.direction[1])
        reward = 0
        self.done = False
        if (0 <= new_head[0] < self.current_size) and (0 <= new_head[1] < self.current_size) and (new_head not in self.snake):
            self.snake = [new_head] + self.snake
            if new_head == self.food:
                reward = 1
                self.food = (random.randint(0,self.current_size-1), random.randint(0,self.current_size-1))
            else:
                self.snake.pop()
        else:
            self.done = True
            reward = -1
        return self.get_obs(), reward, self.done

# =======================
# CONV AUTOENCODER
# =======================
class ConvAE(nn.Module):
    def __init__(self, input_shape=(CONFIG['grid_size'],CONFIG['grid_size'])):
        super().__init__()
        c,h,w = 1, input_shape[0], input_shape[1]
        self.conv1 = nn.Conv2d(1,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.fc = nn.Linear(32*h*w, CONFIG['latent_dim'])
        self.decoder_fc = nn.Linear(CONFIG['latent_dim'], 32*h*w)
        self.deconv1 = nn.Conv2d(32,16,3,padding=1)
        self.deconv2 = nn.Conv2d(16,1,3,padding=1)

    def encoder(self,x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = h.view(h.size(0), -1)
        z = self.fc(h)
        return z

    def decoder(self,z):
        h = F.relu(self.decoder_fc(z))
        h = h.view(h.size(0),32,CONFIG['grid_size'],CONFIG['grid_size'])
        h = F.relu(self.deconv1(h))
        recon = self.deconv2(h)
        return recon

# =======================
# RESONATOR / MEMORY
# =======================
class Resonator:
    def __init__(self, ae):
        self.ae = ae
        self.memory_graph = nx.DiGraph()

    def resonate(self, x, z_prev=None, mem_sample=None):
        z0 = self.ae.encoder(x)
        if z_prev is not None:
            # temporal regularization
            z = z0 + CONFIG['temporal_weight']*(z_prev - z0)
        else:
            z = z0
        # memory update
        idx = len(self.memory_graph)
        self.memory_graph.add_node(idx, latent=z.detach().numpy(), pos=(random.randint(0,CONFIG['grid_size']), random.randint(0,CONFIG['grid_size'])))
        if mem_sample is not None:
            self.memory_graph.add_edge(mem_sample, idx)
        return z

# =======================
# EPISODE / ACTION SELECTION
# =======================
def epistemic_action_selection(env, z, resonator):
    # evaluate free-energy for all actions (placeholder)
    fe = []
    for a in range(4):
        obs_copy = env.get_obs().copy()
        # simple predicted next-head
        hx,hy = env.snake[0]
        dx,dy = [(0,1),(0,-1),(-1,0),(1,0)][a]
        nh = (hx+dx, hy+dy)
        # compute epistemic surprise as distance to nearest memory node
        min_dist = min([np.linalg.norm(np.array(nh)-np.array(resonator.memory_graph.nodes[n]['pos'])) for n in resonator.memory_graph.nodes] + [0])
        fe.append(-min_dist)
    return int(np.argmax(fe))

# =======================
# LIVE VISUALIZATION
# =======================
def render_overlay(env, memory_graph, sigma3_paths=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
        plt.ion()
    ax.clear()
    # snake head & body
    hx, hy = env.snake[0]
    ax.scatter(hx, hy, c='r', s=100, label='head')
    if len(env.snake)>1:
        body = np.array(env.snake[1:])
        ax.scatter(body[:,0], body[:,1], c='b', s=50, label='body')
    # food
    if env.food:
        ax.scatter(env.food[0], env.food[1], c='g', s=100, label='food')
    # memory graph
    pos = {i:(d['pos'][0],d['pos'][1]) for i,d in memory_graph.nodes(data=True)}
    nx.draw_networkx_nodes(memory_graph, pos, node_color='orange', ax=ax, node_size=50)
    nx.draw_networkx_edges(memory_graph, pos, ax=ax, alpha=0.3)
    # sigma3 paths
    if sigma3_paths:
        for path in sigma3_paths:
            path_arr = np.array(path)
            ax.plot(path_arr[:,0], path_arr[:,1], 'm--', linewidth=2, alpha=0.7)
    ax.set_xlim(0, CONFIG['grid_size'])
    ax.set_ylim(0, CONFIG['grid_size'])
    ax.set_title("Snake + Food + Memory Graph + σ₃ Paths")
    ax.grid(True)
    ax.legend(loc='upper right')
    plt.pause(0.001)

# =======================
# MAIN LOOP
# =======================
def main():
    device='cpu'
    env = SnakeEnv()
    ae = ConvAE().to(device)
    resonator = Resonator(ae)

    obs = env.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    z = None

    fig, ax = plt.subplots(figsize=(6,6))
    plt.ion()

    for step in range(CONFIG['rollout_steps']):
        mem_sample = random.choice(list(resonator.memory_graph.nodes)) if resonator.memory_graph.nodes else None
        z = resonator.resonate(obs_tensor, None if z is None else z, mem_sample)

        # select action
        action = epistemic_action_selection(env, z, resonator)
        obs, reward, done = env.step(action)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # generate sigma3 paths (placeholder: simple straight-line future positions)
        sigma3_paths = []
        hx,hy = env.snake[0]
        path = [(hx+i, hy) for i in range(1,5) if hx+i < CONFIG['grid_size']]
        sigma3_paths.append(path)

        # render overlay
        render_overlay(env, resonator.memory_graph, sigma3_paths=sigma3_paths, ax=ax)

        if done:
            print(f"Snake died at step {step}")
            break

if __name__=="__main__":
    main()

