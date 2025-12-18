#!/usr/bin/env python3
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt

# ======================
# CONFIG
# ======================
CONFIG = {
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    "grid": 8,
    "max_steps": 200,

    "latent": 16,
    "ae_lr": 1e-3,
    "ae_epochs": 30,

    "ppo_lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip": 0.2,
    "ppo_epochs": 4,
    "batch": 64,
    "rollout": 512,

    "curiosity_beta": 0.2,
}

device = torch.device(CONFIG["device"])
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
random.seed(CONFIG["seed"])

# ======================
# ENVIRONMENT
# ======================
class SnakeEnv:
    def __init__(self, n):
        self.n = n
        self.reset()

    def reset(self):
        self.snake = [(self.n // 2, self.n // 2)]
        self.dir = random.randrange(4)
        self.place_food()
        self.done = False
        self.steps = 0
        return self.obs()

    def place_food(self):
        empty = [(i, j) for i in range(self.n) for j in range(self.n) if (i, j) not in self.snake]
        self.food = random.choice(empty)

    def obs(self):
        o = np.zeros((3, self.n, self.n), dtype=np.float32)
        for i, (x, y) in enumerate(self.snake):
            o[1 if i else 0, x, y] = 1
        fx, fy = self.food
        o[2, fx, fy] = 1
        return o

    def step(self, a):
        if self.done:
            return self.obs(), 0.0, True, {}

        dx = [(-1,0),(0,1),(1,0),(0,-1)][a]
        hx, hy = self.snake[0]
        nx, ny = hx + dx[0], hy + dx[1]
        self.steps += 1

        if nx < 0 or ny < 0 or nx >= self.n or ny >= self.n or (nx, ny) in self.snake:
            self.done = True
            return self.obs(), -5.0, True, {}

        self.snake.insert(0, (nx, ny))
        reward = -0.01

        if (nx, ny) == self.food:
            reward = 10.0
            self.place_food()
        else:
            self.snake.pop()

        if self.steps >= CONFIG["max_steps"]:
            self.done = True

        return self.obs(), reward, self.done, {}

def obs_tensor(o):
    return torch.tensor(o, dtype=torch.float32).unsqueeze(0).to(device)

# ======================
# AUTOENCODER (SAFE)
# ======================
class Encoder(nn.Module):
    def __init__(self, latent):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1), nn.ReLU()
        )
        with torch.no_grad():
            d = torch.zeros(1,3,CONFIG["grid"],CONFIG["grid"])
            self.flat = self.conv(d).view(1,-1).shape[1]
        self.fc = nn.Linear(self.flat, latent)

    def forward(self,x):
        h = self.conv(x).view(x.size(0),-1)
        return self.fc(h)

class Decoder(nn.Module):
    def __init__(self, flat, latent):
        super().__init__()
        self.fc = nn.Linear(latent, flat)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(32,32,3,padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32,16,3,padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16,3,3,padding=1), nn.Sigmoid()
        )

    def forward(self,z):
        h = self.fc(z).view(-1,32,CONFIG["grid"],CONFIG["grid"])
        return self.deconv(h)

# ======================
# PPO + CURIOSITY
# ======================
class ActorCritic(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim,128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU()
        )
        self.pi = nn.Linear(128,4)
        self.v = nn.Linear(128,1)

    def forward(self,x):
        h = self.net(x)
        return self.pi(h), self.v(h)

class ForwardModel(nn.Module):
    def __init__(self, latent):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent+4,64), nn.ReLU(),
            nn.Linear(64,latent)
        )

    def forward(self,z,a):
        oh = F.one_hot(a,4).float()
        return self.net(torch.cat([z,oh],1))

# ======================
# TRAINING
# ======================
def main():
    env = SnakeEnv(CONFIG["grid"])

    enc = Encoder(CONFIG["latent"]).to(device)
    dec = Decoder(enc.flat, CONFIG["latent"]).to(device)
    ae_opt = optim.Adam(list(enc.parameters())+list(dec.parameters()), lr=CONFIG["ae_lr"])

    # ---- AE PRETRAIN ----
    print("[Phase 1] AE training")
    data=[]
    for _ in range(500):
        o = env.reset()
        for _ in range(20):
            data.append(o)
            o,_,d,_ = env.step(random.randrange(4))
            if d: break

    data = torch.tensor(np.array(data)).to(device)
    for ep in range(CONFIG["ae_epochs"]):
        idx = torch.randperm(len(data))
        loss=0
        for i in range(0,len(data),64):
            b = data[idx[i:i+64]]
            z = enc(b)
            r = dec(z)
            l = F.mse_loss(r,b)
            ae_opt.zero_grad(); l.backward(); ae_opt.step()
            loss+=l.item()
        if (ep+1)%5==0:
            print(f"  AE Epoch {ep+1} loss {loss/len(data):.4f}")

    # ---- PPO ----
    ac = ActorCritic(CONFIG["latent"]).to(device)
    fwd = ForwardModel(CONFIG["latent"]).to(device)
    opt = optim.Adam(ac.parameters(), lr=CONFIG["ppo_lr"])
    opt_fwd = optim.Adam(fwd.parameters(), lr=1e-3)

    print("[Phase 2] PPO Training")
    rewards_log=[]

    for update in range(1,201):
        obs = env.reset()
        buf = []

        for _ in range(CONFIG["rollout"]):
            z = enc(obs_tensor(obs))
            logits, val = ac(z)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()

            next_obs, r, d, _ = env.step(a.item())
            z2 = enc(obs_tensor(next_obs)).detach()

            pred = fwd(z.detach(), a)
            curiosity = F.mse_loss(pred, z2).item()
            r += CONFIG["curiosity_beta"] * curiosity

            buf.append((z.detach(),a.detach(),dist.log_prob(a).detach(),r,val.detach(),d))
            opt_fwd.zero_grad()
            F.mse_loss(pred,z2).backward()
            opt_fwd.step()

            obs = next_obs if not d else env.reset()

        # ---- PPO UPDATE ----
        Z,A,LP,R,V,D = zip(*buf)
        returns=[]
        adv=[]
        gae=0
        next_v=0

        for i in reversed(range(len(R))):
            delta = R[i] + CONFIG["gamma"]*next_v*(1-D[i]) - V[i].item()
            gae = delta + CONFIG["gamma"]*CONFIG["gae_lambda"]*gae*(1-D[i])
            adv.insert(0,gae)
            returns.insert(0,gae+V[i].item())
            next_v = V[i].item()

        Z=torch.cat(Z)
        A=torch.cat(A)
        LP=torch.cat(LP)
        ADV=torch.tensor(adv).to(device)
        RET=torch.tensor(returns).to(device)

        for _ in range(CONFIG["ppo_epochs"]):
            idx = torch.randperm(len(Z))
            for i in range(0,len(Z),CONFIG["batch"]):
                j=idx[i:i+CONFIG["batch"]]
                logits,v = ac(Z[j])
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(A[j])
                ratio = (lp-LP[j]).exp()

                surr = torch.min(
                    ratio*ADV[j],
                    torch.clamp(ratio,1-CONFIG["clip"],1+CONFIG["clip"])*ADV[j]
                )
                loss = -surr.mean() + 0.5*F.mse_loss(v.squeeze(),RET[j]) - 0.01*dist.entropy().mean()
                opt.zero_grad(); loss.backward(); opt.step()

        avg = np.mean([sum(r for *_,r,_,_ in buf)])
        rewards_log.append(avg)
        if update%10==0:
            print(f"  Update {update} avg reward {np.mean(rewards_log[-10:]):.2f}")

    plt.plot(rewards_log)
    plt.title("Training Reward")
    plt.show()

if __name__=="__main__":
    main()

