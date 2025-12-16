#!/usr/bin/env python3
"""
NEURO-SYMBOLIC COGNITIVE AGENT (NSCA)
=====================================
A unified architecture combining Autoencoders, Resonant Optimization,
Symbolic Clustering, and Proximal Policy Optimization (PPO).

Architecture Components:
1. Perception: Autoencoder for state representation learning.
2. Inference: Gradient-based latent space optimization (Resonance).
3. Memory: Coherence-gated experience buffer.
4. Abstraction: K-Means clustering for symbol discovery.
5. Control: PPO Agent with Intrinsic Motivation (Curiosity).
6. Refinement: Offline analysis of high-variance memories.

Author: Generated for User
Date: 2025-12-09
"""

import os
import random
import time
import math
from pathlib import Path
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'seed': 42,
    'device': 'cpu',  # Set to 'cuda' if GPU is available
    'output_directory': 'nsca_output',
    
    # --- Environment Settings ---
    'grid_size': 8,
    'max_episode_steps': 200,

    # --- Phase 1: Representation Learning (Perception) ---
    'pretrain_samples': 2500,
    'autoencoder_epochs': 100,
    'autoencoder_lr': 1e-3,
    'latent_dimension': 16,

    # --- Phase 2: Exploratory Data Collection ---
    'exploration_episodes': 30, 
    'inference_optimization_steps': 8,  # Steps for gradient-based refinement
    'inference_lr': 0.08,
    'memory_coherence_threshold': 0.05, # Max reconstruction loss to accept memory
    'memory_buffer_size': 5000,

    # --- Phase 3: Symbolic Abstraction ---
    'initial_cluster_count': 4, 

    # --- Phase 4: Reinforcement Learning (PPO) ---
    'ppo_updates': 100, 
    'ppo_learning_rate': 3e-4,
    'rollout_buffer_size': 1024,
    'curiosity_weight': 0.5, 

    # --- Phase 5: Refinement (Offline Analysis) ---
    'outlier_threshold': 0.5, 
}

# Setup Directories and Device
OUTPUT_DIR = Path(CONFIG['output_directory'])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device(CONFIG['device'])

# Set Random Seeds for Reproducibility
random.seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])

# ==========================================
# ENVIRONMENT (Snake Grid)
# ==========================================
class SnakeEnvironment:
    """
    A simple grid-based environment where an agent seeks targets (food).
    Returns a 3-channel grid observation.
    """
    def __init__(self, size=8):
        self.size = size
        self.reset()

    def reset(self):
        self.snake = [(self.size // 2, self.size // 2)]
        self.place_target()
        self.done = False
        self.score = 0
        self.steps = 0
        return self._get_observation()

    def place_target(self):
        empty_cells = [(x, y) for x in range(self.size) for y in range(self.size) if (x, y) not in self.snake]
        self.target = random.choice(empty_cells) if empty_cells else None

    def step(self, action):
        # Actions: 0=Up, 1=Right, 2=Down, 3=Left
        dx, dy = [(-1, 0), (0, 1), (1, 0), (0, -1)][action]
        head_x, head_y = self.snake[0]
        new_x, new_y = head_x + dx, head_y + dy
        self.steps += 1

        # Check Collisions (Walls or Self)
        if new_x < 0 or new_y < 0 or new_x >= self.size or new_y >= self.size or (new_x, new_y) in self.snake:
            self.done = True
            return self._get_observation(), -10.0, True, {}

        self.snake.insert(0, (new_x, new_y))
        reward = 0.0
        
        # Check Target Reached
        if self.target and (new_x, new_y) == self.target:
            reward = 10.0
            self.score += 1
            self.place_target()
        else:
            self.snake.pop() # Move tail

        # Truncate episode if too long
        if self.steps >= CONFIG['max_episode_steps']:
            self.done = True
            
        return self._get_observation(), reward, self.done, {}

    def _get_observation(self):
        # Channels: [Snake Body, Empty, Target] - Simplified for this example
        obs = np.zeros((3, self.size, self.size), dtype=np.float32)
        head_x, head_y = self.snake[0]
        obs[0, head_x, head_y] = 1.0 # Head
        for x, y in self.snake[1:]:
            obs[1, x, y] = 1.0 # Body
        if self.target:
            tx, ty = self.target
            obs[2, tx, ty] = 1.0 # Target
        return obs

    def copy_state(self):
        return {'snake': list(self.snake), 'target': self.target, 'done': self.done}

    def set_state(self, state):
        self.snake = list(state['snake'])
        self.target = state['target']
        self.done = state['done']

    def get_manhattan_distance(self):
        if not self.target: return 0
        head_x, head_y = self.snake[0]
        target_x, target_y = self.target
        return abs(head_x - target_x) + abs(head_y - target_y)

# Utilities
def flatten_observation(obs):
    return np.array(obs, dtype=np.float32).flatten()

def observation_to_tensor(obs):
    return torch.tensor(flatten_observation(obs), dtype=torch.float32).unsqueeze(0).to(device)

# ==========================================
# MODULE 1: PERCEPTUAL AUTOENCODER
# ==========================================
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, input_dim), nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return z, reconstruction

# ==========================================
# MODULE 2: INFERENCE OPTIMIZER (Resonance)
# ==========================================
class InferenceOptimizer:
    """
    Performs inference-time optimization.
    It adjusts the latent vector 'z' to minimize reconstruction loss against the input 'x'.
    """
    def __init__(self, model):
        self.model = model
        self.criterion = nn.MSELoss()

    def optimize(self, x, steps=CONFIG['inference_optimization_steps'], lr=CONFIG['inference_lr']):
        with torch.no_grad():
            z_initial = self.model.encoder(x)
        
        # Detach z and enable gradients for optimization
        z = z_initial.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([z], lr=lr)
        
        final_loss = 0
        for _ in range(steps):
            optimizer.zero_grad()
            reconstruction = self.model.decoder(z)
            loss = self.criterion(reconstruction, x)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())
            
        return z.detach(), final_loss

# ==========================================
# MODULE 3: GATED MEMORY BUFFER
# ==========================================
class ExperienceMemory:
    def __init__(self, capacity):
        self.vectors = deque(maxlen=capacity)
        self.losses = deque(maxlen=capacity)

    def add(self, z, reconstruction_loss):
        # Gate: Only store experiences where the model is confident (low loss)
        if reconstruction_loss < CONFIG['memory_coherence_threshold']:
            self.vectors.append(z.detach().cpu().numpy().reshape(-1))
            self.losses.append(reconstruction_loss)
            return True
        return False

    def get_all_vectors(self):
        if len(self.vectors) > 0:
            return np.vstack(list(self.vectors))
        return np.zeros((0, CONFIG['latent_dimension']))

# ==========================================
# MODULE 4: ACTOR-CRITIC AGENT (PPO)
# ==========================================
class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, action_dim=4):
        super().__init__()
        # Shared feature extractor
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        # Policy Head (Actor)
        self.policy_head = nn.Sequential(nn.Linear(128, action_dim))
        # Value Head (Critic)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        features = self.shared_layers(x)
        return self.policy_head(features), self.value_head(features).squeeze(-1)

class ForwardPredictor(nn.Module):
    """
    Used for Intrinsic Motivation (Curiosity).
    Predicts next latent state given current state + action.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# ==========================================
# EXECUTION PHASES
# ==========================================

def phase_1_pretraining(env):
    """
    Collects random samples to train the Autoencoder.
    Establishes the initial perceptual representation.
    """
    print("\n--- PHASE 1: REPRESENTATION LEARNING (Autoencoder Training) ---")
    print(f"Collecting {CONFIG['pretrain_samples']} samples via random walk...")
    
    data_buffer = []
    for _ in range(CONFIG['pretrain_samples']):
        obs = env.reset()
        for _ in range(30):
            obs, _, done, _ = env.step(random.randrange(4))
            data_buffer.append(flatten_observation(obs))
            if done: break
    
    training_data = np.vstack(data_buffer)
    input_dim = training_data.shape[1]
    
    # Initialize Models
    autoencoder = Autoencoder(input_dim, CONFIG['latent_dimension']).to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=CONFIG['autoencoder_lr'])
    criterion = nn.MSELoss()
    tensor_data = torch.tensor(training_data, dtype=torch.float32).to(device)
    
    print(f"Training Autoencoder for {CONFIG['autoencoder_epochs']} epochs...")
    for epoch in range(CONFIG['autoencoder_epochs']):
        optimizer.zero_grad()
        z, reconstruction = autoencoder(tensor_data)
        loss = criterion(reconstruction, tensor_data)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Reconstruction Loss = {loss.item():.6f}")
    
    return autoencoder, input_dim

def phase_2_exploration(env, autoencoder, optimizer_module, memory):
    """
    Agent explores the environment using a Model-Based lookahead.
    It simulates actions and selects the one resulting in the lowest reconstruction loss (highest confidence).
    """
    print("\n--- PHASE 2: EXPLORATORY DATA COLLECTION (Model-Based) ---")
    print(f"Collecting experiences. Gating Threshold: {CONFIG['memory_coherence_threshold']}")
    
    for episode in range(1, CONFIG['exploration_episodes'] + 1):
        obs = env.reset()
        episode_reward = 0
        
        while True:
            # Lookahead Planning
            best_action = 0
            best_score = -float('inf')
            current_state_snapshot = env.copy_state()
            distance_start = env.get_manhattan_distance()
            
            for action_candidate in range(4):
                # Simulate Action
                sim_env = SnakeEnvironment(env.size)
                sim_env.set_state(current_state_snapshot)
                sim_obs, sim_reward, sim_done, _ = sim_env.step(action_candidate)
                
                # Check Confidence (Reconstruction Loss)
                obs_tensor = observation_to_tensor(sim_obs)
                _, reconstruction_loss = optimizer_module.optimize(obs_tensor)
                
                # Scoring Function: Minimize Loss + Maximize Reward
                score = -reconstruction_loss
                if sim_reward > 0: score += 10.0
                if sim_done: score -= 10.0
                
                distance_end = sim_env.get_manhattan_distance()
                if distance_end < distance_start: score += 0.5 # Shaping
                
                if score > best_score:
                    best_score = score
                    best_action = action_candidate
            
            # Execute Best Action
            obs, reward, done, _ = env.step(best_action)
            episode_reward += reward
            
            # Store Result in Memory
            obs_tensor = observation_to_tensor(obs)
            z_optimized, loss_optimized = optimizer_module.optimize(obs_tensor)
            memory.add(z_optimized, loss_optimized)
            
            if done: break
            
        if episode % 5 == 0:
            print(f"Episode {episode}: Reward={episode_reward:.1f}, Stored Memories={len(memory.vectors)}")

def phase_3_abstraction(memory):
    """
    Clusters the collected latent vectors to discover discrete states (Symbols).
    """
    print("\n--- PHASE 3: SYMBOLIC ABSTRACTION (Clustering) ---")
    
    memory_vectors = memory.get_all_vectors()
    
    # Safety Check: Ensure enough data points exist
    if len(memory_vectors) < CONFIG['initial_cluster_count']:
        print("WARNING: Insufficient memory data. Initializing random cluster centers.")
        centroids = np.random.randn(CONFIG['initial_cluster_count'], CONFIG['latent_dimension']).astype(np.float32)
    else:
        print(f"Clustering {len(memory_vectors)} memory vectors...")
        kmeans = KMeans(n_clusters=CONFIG['initial_cluster_count'], n_init=10)
        kmeans.fit(memory_vectors)
        centroids = kmeans.cluster_centers_
        
    print(f"Discovered {len(centroids)} symbolic centroids.")
    return centroids

def phase_4_reinforcement_learning(env, autoencoder, optimizer_module, centroids):
    """
    Trains a PPO agent.
    Input State = [Optimized Latent Vector + Symbol Embedding].
    Uses Intrinsic Motivation (Curiosity) to aid exploration.
    """
    print("\n--- PHASE 4: REINFORCEMENT LEARNING (PPO + Curiosity) ---")
    
    num_symbols = len(centroids)
    embedding_dim = 8
    symbol_embeddings = nn.Embedding(num_symbols, embedding_dim).to(device)
    
    # Policy Network Input: Latent Dim (16) + Embedding Dim (8)
    policy_net = ActorCriticNetwork(CONFIG['latent_dimension'] + embedding_dim, action_dim=4).to(device)
    ppo_optimizer = optim.Adam(policy_net.parameters(), lr=CONFIG['ppo_learning_rate'])
    
    # Forward Predictor Input: Latent Dim (16) + Action One-Hot (4)
    forward_predictor = ForwardPredictor(CONFIG['latent_dimension'] + 4, CONFIG['latent_dimension']).to(device)
    predictor_optimizer = optim.Adam(forward_predictor.parameters(), lr=1e-3)
    
    def construct_state(obs):
        # Convert Obs -> Latent -> Optimized Latent
        x = observation_to_tensor(obs)
        z, _ = optimizer_module.optimize(x)
        z_numpy = z.detach().cpu().numpy().reshape(-1)
        
        # Identify nearest symbol
        distances = np.sum((centroids - z_numpy)**2, axis=1)
        symbol_id = np.argmin(distances)
        
        # Create State Tensor
        sym_idx_tensor = torch.tensor([symbol_id], dtype=torch.long).to(device)
        emb_vector = symbol_embeddings(sym_idx_tensor).squeeze(0)
        
        # Concatenate: [Latent, Symbol_Embedding]
        return torch.cat([z.squeeze(0), emb_vector], dim=0), z, symbol_id

    # Training Loop
    for update in range(1, CONFIG['ppo_updates'] + 1):
        obs = env.reset()
        buffer_states, buffer_actions, buffer_logprobs = [], [], []
        buffer_rewards, buffer_dones = [], []
        
        # Collect Rollout
        for _ in range(CONFIG['rollout_buffer_size']):
            current_state, z_current, _ = construct_state(obs)
            
            # Action Selection
            logits, _ = policy_net(current_state.unsqueeze(0))
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            
            # Step Environment
            obs_next, extrinsic_reward, done, _ = env.step(action.item())
            
            # Curiosity Calculation
            _, z_next, _ = construct_state(obs_next)
            
            # Prepare Input for Predictor: [z_current, action_one_hot]
            action_one_hot = torch.zeros(4).to(device)
            action_one_hot[action.item()] = 1.0
            
            predictor_input = torch.cat([z_current.squeeze(0), action_one_hot], dim=0)
            z_predicted = forward_predictor(predictor_input)
            
            # Intrinsic Reward = Prediction Error (Surprise)
            intrinsic_reward = ((z_predicted - z_next.squeeze(0))**2).mean().item()
            
            total_reward = extrinsic_reward + (CONFIG['curiosity_weight'] * intrinsic_reward)
            
            # Store Buffer
            buffer_states.append(current_state)
            buffer_actions.append(action)
            buffer_logprobs.append(dist.log_prob(action))
            buffer_rewards.append(total_reward)
            buffer_dones.append(done)
            
            # Train Predictor Online
            predictor_loss = ((z_predicted - z_next.squeeze(0).detach())**2).mean()
            predictor_optimizer.zero_grad()
            predictor_loss.backward()
            predictor_optimizer.step()
            
            obs = obs_next
            if done: obs = env.reset()
            
        # PPO Update
        states_tensor = torch.stack(buffer_states).detach()
        actions_tensor = torch.stack(buffer_actions).detach()
        old_logprobs_tensor = torch.stack(buffer_logprobs).detach()
        
        # Calculate Returns (Discounted Reward)
        returns = []
        Gt = 0
        for r, d in zip(reversed(buffer_rewards), reversed(buffer_dones)):
            Gt = r + 0.99 * Gt * (1 - int(d))
            returns.insert(0, Gt)
        returns_tensor = torch.tensor(returns, dtype=torch.float32).to(device)
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        
        # Optimization Epochs
        for _ in range(4):
            logits, values = policy_net(states_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            new_logprobs = dist.log_prob(actions_tensor)
            entropy = dist.entropy().mean()
            
            ratio = torch.exp(new_logprobs - old_logprobs_tensor)
            advantage = returns_tensor - values.squeeze()
            
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantage
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * ((returns_tensor - values.squeeze())**2).mean()
            
            loss = actor_loss + critic_loss - (0.01 * entropy)
            
            ppo_optimizer.zero_grad()
            loss.backward()
            ppo_optimizer.step()
            
        if update % 10 == 0:
            avg_reward = sum(buffer_rewards) / (sum(buffer_dones) + 1)
            print(f"Update {update}/{CONFIG['ppo_updates']}: Average Reward per Episode = {avg_reward:.2f}")

def phase_5_refinement(memory, centroids):
    """
    Offline analysis. Detects memories that do not fit well into existing clusters.
    If variance is high, synthesizes a new centroid (symbol) for future use.
    """
    print("\n--- PHASE 5: OFFLINE REFINEMENT (Outlier Analysis) ---")
    
    memory_vectors = memory.get_all_vectors()
    
    if len(memory_vectors) < 10:
        print("Insufficient data for analysis.")
        return centroids

    # Calculate distances to nearest centroids
    distances = np.zeros(len(memory_vectors))
    for i, vec in enumerate(memory_vectors):
        dists = np.sum((centroids - vec)**2, axis=1)
        distances[i] = np.min(dists)
        
    avg_distance = np.mean(distances)
    outliers = memory_vectors[distances > avg_distance * 1.5]
    
    print(f"Analysis complete. Found {len(outliers)} outliers (contradictory states).")
    
    if len(outliers) > 20:
        print("Refining Knowledge Base: Synthesizing NEW Symbol from outliers...")
        kmeans_refinement = KMeans(n_clusters=1)
        kmeans_refinement.fit(outliers)
        new_symbol = kmeans_refinement.cluster_centers_
        
        updated_centroids = np.vstack([centroids, new_symbol])
        print(f"Refinement Successful. Symbol Count: {len(centroids)} -> {len(updated_centroids)}")
        return updated_centroids
    else:
        print("Knowledge Base is stable. No new symbols required.")
        return centroids

# ==========================================
# MAIN CONTROL LOOP
# ==========================================
def main():
    print("=== NEURO-SYMBOLIC COGNITIVE AGENT (NSCA) INITIALIZED ===")
    
    # Initialize Environment
    env = SnakeEnvironment(CONFIG['grid_size'])
    
    # Phase 1: Pre-training (Vision)
    autoencoder, input_dim = phase_1_pretraining(env)
    
    # Initialize Optimization & Memory Modules
    optimizer_module = InferenceOptimizer(autoencoder)
    memory = ExperienceMemory(CONFIG['memory_buffer_size'])
    
    # Phase 2: Exploration (Data Collection)
    phase_2_exploration(env, autoencoder, optimizer_module, memory)
    
    # Phase 3: Abstraction (Symbol Discovery)
    centroids = phase_3_abstraction(memory)
    
    # Phase 4: Policy Learning (PPO)
    phase_4_reinforcement_learning(env, autoencoder, optimizer_module, centroids)
    
    # Phase 5: Refinement (Evolution)
    final_centroids = phase_5_refinement(memory, centroids)
    
    # Save State
    print("\nProcess Complete. Saving Final State...")
    np.save(OUTPUT_DIR / "final_symbol_centroids.npy", final_centroids)
    print(f"Artifacts saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
