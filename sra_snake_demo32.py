import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# -----------------------------
# CONFIG
# -----------------------------
CONFIG = {
    'latent_dim': 64,
    'temporal_weight': 0.1,
    'fwd_weight': 0.1,
    'lr': 1e-3,
    'max_steps': 200,
    'grid_size': 10,
    'num_actions': 4,
}

device = torch.device('cpu')

# -----------------------------
# SIMPLE CONV AE
# -----------------------------
class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16*(CONFIG['grid_size']//4)**2, CONFIG['latent_dim'])
        )
        self.decoder = nn.Sequential(
            nn.Linear(CONFIG['latent_dim'], 16*(CONFIG['grid_size']//4)**2),
            nn.ReLU(),
            nn.Unflatten(1, (16, CONFIG['grid_size']//4, CONFIG['grid_size']//4)),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8,1,3,stride=2,padding=1,output_padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon

# -----------------------------
# FORWARD MODEL
# -----------------------------
class ForwardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(CONFIG['latent_dim'] + CONFIG['num_actions'], 128),
            nn.ReLU(),
            nn.Linear(128, CONFIG['latent_dim'])
        )
    def forward(self, z, a):
        x = torch.cat([z,a],dim=-1)
        return self.fc(x)

# -----------------------------
# SIMPLE SNAKE ENV
# -----------------------------
class SnakeEnv:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.reset()
    def reset(self):
        self.snake = deque([[self.grid_size//2,self.grid_size//2]])
        self.food = [np.random.randint(self.grid_size), np.random.randint(self.grid_size)]
        self.done = False
        return self._get_obs()
    def _get_obs(self):
        obs = np.zeros((self.grid_size,self.grid_size))
        for x,y in self.snake:
            obs[x,y] = 1
        fx, fy = self.food
        obs[fx,fy] = 2
        return obs[np.newaxis,:,:].astype(np.float32)  # 1xHxW
    def step(self, action):
        if self.done: return self._get_obs(),0,self.done
        head = self.snake[-1].copy()
        if action==0: head[0]-=1
        elif action==1: head[0]+=1
        elif action==2: head[1]-=1
        elif action==3: head[1]+=1
        # bounds
        if head[0]<0 or head[0]>=self.grid_size or head[1]<0 or head[1]>=self.grid_size or head in self.snake:
            self.done=True
            return self._get_obs(), -1, self.done
        self.snake.append(head)
        reward = 0
        if head==self.food:
            reward=1
            self.food = [np.random.randint(self.grid_size), np.random.randint(self.grid_size)]
        else:
            self.snake.popleft()
        return self._get_obs(), reward, self.done

# -----------------------------
# ACTION SELECTION (EPISODIC FREE-ENERGY)
# -----------------------------
def onehot_action(a):
    oh = torch.zeros(CONFIG['num_actions'])
    oh[a]=1
    return oh

def select_action(env, ae, z, fwd_model, steps_ahead=3):
    scores = []
    paths=[]
    for a in range(CONFIG['num_actions']):
        path=[]
        z_curr = z.detach()
        a_oh = onehot_action(a).unsqueeze(0)
        for _ in range(steps_ahead):
            z_next = fwd_model(z_curr, a_oh)
            path.append(z_next.squeeze(0).detach().numpy())
            z_curr = z_next.detach()
        score = -torch.norm(z_next)  # simple epistemic value: smaller latent = preferred
        scores.append(score.item())
        paths.append(path)
    best_idx = int(np.argmax(scores))
    return best_idx, [paths[best_idx]] if paths else []

# -----------------------------
# TRAINING LOOP WITH LIVE VISUALIZATION
# -----------------------------
def run_game():
    env = SnakeEnv(CONFIG['grid_size'])
    ae = ConvAE().to(device)
    fwd_model = ForwardModel().to(device)
    optimizer = torch.optim.Adam(list(ae.parameters())+list(fwd_model.parameters()), lr=CONFIG['lr'])

    plt.ion()
    fig,ax = plt.subplots()

    for episode in range(5):
        obs = env.reset()
        z_prev = None
        total_reward = 0

        for step in range(CONFIG['max_steps']):
            obs_tensor = torch.tensor(obs).unsqueeze(0).to(device)
            z, recon = ae(obs_tensor)

            # select action
            a, paths = select_action(env, ae, z, fwd_model)
            obs_next, reward, done = env.step(a)
            obs_next_tensor = torch.tensor(obs_next).unsqueeze(0).to(device)
            z_next, _ = ae(obs_next_tensor)

            # LOSSES
            loss_rec = F.mse_loss(recon, obs_tensor)
            if z_prev is None:
                loss_temp = 0
            else:
                loss_temp = CONFIG['temporal_weight'] * F.mse_loss(z, z_prev.detach())
            a_oh = onehot_action(a).unsqueeze(0)
            loss_fwd = CONFIG['fwd_weight'] * F.mse_loss(fwd_model(z.detach(),a_oh), z_next.detach())
            loss = loss_rec + loss_temp + loss_fwd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_reward += reward
            z_prev = z.detach()
            obs = obs_next

            # ---- LIVE VISUALIZATION ----
            ax.clear()
            grid = obs_next[0]
            ax.imshow(grid, cmap='gray', vmin=0, vmax=2)
            # plot predicted future path
            if paths:
                for path in paths:
                    coords = path[0]  # latent; could add decoder mapping to spatial grid
                    ax.plot([CONFIG['grid_size']//2]*len(path), np.arange(len(path)), 'r.-')
            ax.set_title(f"EpSRA Snake | Ep {episode} Step {step} | Reward {total_reward}")
            plt.pause(0.01)
            if done: break
    plt.ioff()
    plt.show()

# -----------------------------
# MAIN
# -----------------------------
if __name__=="__main__":
    run_game()

