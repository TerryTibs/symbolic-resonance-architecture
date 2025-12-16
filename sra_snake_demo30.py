import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# CONFIG
# -------------------------
CONFIG = {
    'grid_size': 10,
    'latent_dim': 64,
    'max_steps': 200,
    'temporal_weight': 0.1,
    'device': 'cpu',
    'lookahead': 5,  # multi-step prediction
    'prune_threshold': 0.5,  # epistemic pruning
}

# -------------------------
# ENVIRONMENT
# -------------------------
class SnakeEnv:
    def __init__(self, grid_size=CONFIG['grid_size']):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.snake = deque([[self.grid_size // 2, self.grid_size // 2]])
        self.direction = [0, 1]
        self.food = self._place_food()
        self.steps = 0
        return self._get_obs()

    def _place_food(self):
        while True:
            f = [np.random.randint(0, self.grid_size),
                 np.random.randint(0, self.grid_size)]
            if f not in self.snake:
                return f

    def _get_obs(self):
        grid = np.zeros((self.grid_size, self.grid_size))
        for x, y in self.snake:
            grid[x, y] = 1
        fx, fy = self.food
        grid[fx, fy] = 2
        return grid

    def step(self, action):
        if action == 0: self.direction = [-1, 0]
        elif action == 1: self.direction = [1, 0]
        elif action == 2: self.direction = [0, -1]
        elif action == 3: self.direction = [0, 1]

        new_head = [self.snake[0][0] + self.direction[0],
                    self.snake[0][1] + self.direction[1]]
        reward = 0
        done = False

        # Collisions
        if (new_head in list(self.snake) or
            not all(0 <= c < self.grid_size for c in new_head)):
            done = True
            reward = -1
            return self._get_obs(), reward, done

        self.snake.appendleft(new_head)

        # Eating food
        if new_head == self.food:
            reward = 1
            self.food = self._place_food()  # new food
        else:
            self.snake.pop()  # normal move

        self.steps += 1
        if self.steps >= CONFIG['max_steps']:
            done = True

        return self._get_obs(), reward, done

# -------------------------
# CONV AUTOENCODER
# -------------------------
class ConvAE(nn.Module):
    def __init__(self, latent_dim=CONFIG['latent_dim'], grid_size=CONFIG['grid_size']):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.grid_size = grid_size
        self.fc = nn.Linear(32 * grid_size * grid_size, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, 32 * grid_size * grid_size)
        self.deconv1 = nn.ConvTranspose2d(32, 16, 3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(16, 1, 3, padding=1)

    def encoder(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = h.view(h.size(0), -1)
        z = self.fc(h)
        return z

    def decoder(self, z):
        h = self.decoder_fc(z)
        h = h.view(-1, 32, self.grid_size, self.grid_size)
        h = F.relu(self.deconv1(h))
        h = torch.sigmoid(self.deconv2(h))
        return h

# -------------------------
# MEMORY GRAPH
# -------------------------
class MemoryGraph:
    def __init__(self):
        self.G = nx.DiGraph()

    def add_state(self, pos):
        self.G.add_node(tuple(pos))

    def add_transition(self, from_pos, to_pos):
        self.G.add_edge(tuple(from_pos), tuple(to_pos))

# -------------------------
# RESONATOR
# -------------------------
class Resonator:
    def __init__(self, ae):
        self.ae = ae

    def resonate(self, x):
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            z = self.ae.encoder(x_tensor)
        return z

# -------------------------
# MULTI-STEP σ₃ PREDICTIONS
# -------------------------
def predict_sigma3_paths(env, resonator, lookahead=CONFIG['lookahead']):
    head = env.snake[0]
    possible_paths = []
    directions = [[-1,0],[1,0],[0,-1],[0,1]]

    for d in directions:
        path = [head.copy()]
        pos = head.copy()
        for _ in range(lookahead):
            pos = [pos[0]+d[0], pos[1]+d[1]]
            # Check bounds
            if not all(0 <= c < CONFIG['grid_size'] for c in pos) or pos in path or pos in env.snake:
                break
            path.append(pos.copy())
        if len(path) > 1:
            possible_paths.append(path)
    # Epistemic pruning: remove short paths
    pruned = [p for p in possible_paths if len(p)/lookahead >= CONFIG['prune_threshold']]
    return pruned

# -------------------------
# EPSTEMIC ACTION SELECTION
# -------------------------
def select_action(env, resonator):
    paths = predict_sigma3_paths(env, resonator)
    if not paths:
        return np.random.randint(0,4)
    # Choose path that moves closest to food
    scores = [ -np.linalg.norm(np.array(p[-1]) - np.array(env.food)) for p in paths ]
    best_path = paths[np.argmax(scores)]
    # Determine direction for first step
    head = env.snake[0]
    next_step = best_path[1]
    delta = [next_step[0]-head[0], next_step[1]-head[1]]
    dir_map = {(-1,0):0, (1,0):1, (0,-1):2, (0,1):3}
    return dir_map.get(tuple(delta), np.random.randint(0,4)), paths

# -------------------------
# LIVE PLOTTING
# -------------------------
def plot_state(env, G=None, paths=None):
    plt.clf()
    grid = env._get_obs()
    plt.imshow(grid, cmap='gray_r', vmin=0, vmax=2)
    fx, fy = env.food
    plt.scatter(fy, fx, color='green', s=100, label='food')
    # Snake
    sx=[p[1] for p in env.snake]
    sy=[p[0] for p in env.snake]
    plt.plot(sx,sy,'b-', linewidth=2)
    # Memory graph
    if G:
        pos = {n:(n[1], CONFIG['grid_size']-1-n[0]) for n in G.nodes()}
        nx.draw(G, pos, node_size=50, node_color='blue', alpha=0.3)
    # Predicted σ₃ paths
    if paths:
        for path in paths:
            xs=[p[1] for p in path]
            ys=[p[0] for p in path]
            plt.plot(xs, ys, 'r--', alpha=0.5)
    plt.pause(0.05)

# -------------------------
# MAIN GAME LOOP
# -------------------------
def run_game():
    env = SnakeEnv()
    ae = ConvAE()
    resonator = Resonator(ae)
    mem = MemoryGraph()
    plt.ion()
    obs = env.reset()
    done = False

    while True:
        mem.add_state(env.snake[0])
        action, paths = select_action(env,resonator)
        obs, reward, done = env.step(action)
        mem.add_transition(env.snake[0], env.snake[0])
        plot_state(env, mem.G, paths)
        if done:
            obs = env.reset()
            mem = MemoryGraph()

    plt.ioff()
    plt.show()

if __name__=='__main__':
    run_game()

