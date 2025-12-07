import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import numpy as np

# ==========================================
# THE SYMBOLIC RESONANCE ARCHITECTURE (SRA)
# Core Logic Kernel
# ==========================================

# CONFIGURATION
CONFIG = {
    'latent_dim': 12,
    'memory_threshold': 0.02,
    'resonance_steps': 5,
}

# STAGE 1: PERCEPTUAL CORE (The Senses)
class PerceptualCore(nn.Module):
    def __init__(self):
        super(PerceptualCore, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, CONFIG['latent_dim'])
        )
        self.decoder = nn.Sequential(
            nn.Linear(CONFIG['latent_dim'], 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 28 * 28), nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)

# STAGE 2: RESONANCE ENGINE (The Stability Loop)
class ResonantEngine:
    def __init__(self, model):
        self.model = model
        self.criterion = nn.MSELoss()

    def resonate(self, x, steps=CONFIG['resonance_steps']):
        # "Meditate" on the input to find the most stable representation
        with torch.no_grad():
            z_initial = self.model.encoder(x)
        
        z = z_initial.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([z], lr=0.1)

        for _ in range(steps):
            optimizer.zero_grad()
            recon = self.model.decoder(z)
            loss = self.criterion(recon, x)
            loss.backward()
            optimizer.step()
        
        return z.detach(), loss.item()

# STAGE 3: GATED MEMORY (The Knowledge Graph)
class CognitiveMemory:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.counter = 0

    def add_event(self, latent_vector, loss):
        # Only store thoughts that are "Clear" (Low Loss)
        if loss < CONFIG['memory_threshold']:
            node_id = f"mem_{self.counter}"
            self.graph.add_node(node_id, vector=latent_vector, coherence=loss)
            
            if self.counter > 0:
                prev_node = f"mem_{self.counter - 1}"
                self.graph.add_edge(prev_node, node_id)
            
            self.counter += 1
            return True
        return False

# EXECUTION HARNESS
if __name__ == "__main__":
    print("Initializing SRA Core Kernel...")
    core = PerceptualCore()
    engine = ResonantEngine(core)
    memory = CognitiveMemory()
    print("SRA Kernel Online. Ready for Input.")
