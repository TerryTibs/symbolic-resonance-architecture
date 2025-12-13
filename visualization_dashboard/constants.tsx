import React from 'react';
import { RoadmapStageData, ModuleStatus } from './types';

const EyeIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-teal-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M2.036 12.322a1.012 1.012 0 010-.639l4.418-6.312a1.012 1.012 0 011.583 0l4.418 6.312a1.012 1.012 0 010 .639l-4.418 6.312a1.012 1.012 0 01-1.583 0l-4.418-6.312z" />
    <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
  </svg>
);

const LinkIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-lime-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M13.19 8.688a4.5 4.5 0 011.242 7.244l-4.5 4.5a4.5 4.5 0 01-6.364-6.364l1.757-1.757m13.35-.622l1.757-1.757a4.5 4.5 0 00-6.364-6.364l-4.5 4.5a4.5 4.5 0 001.242 7.244" />
    </svg>
);

const BrainCircuitIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 3v1.5M4.5 8.25H3m18 0h-1.5M4.5 12H3m18 0h-1.5m-15 3.75H3m18 0h-1.5M8.25 21v-1.5M12 3.75h.008v.008H12V3.75zm-3.75 0h.008v.008H8.25V3.75zm0 3.75h.008v.008H8.25V7.5zm0 3.75h.008v.008H8.25v-3.75zm0 3.75h.008v.008H8.25v-3.75zm0 3.75h.008v.008H8.25v-3.75zm3.75 3.75h.008v.008H12v-3.75zm0-3.75h.008v.008H12v-3.75zm0-3.75h.008v.008H12V7.5zm0-3.75h.008v.008H12V3.75zm3.75 0h.008v.008h-.008V3.75zm0 3.75h.008v.008h-.008V7.5zm0 3.75h.008v.008h-.008v-3.75zm0 3.75h.008v.008h-.008v-3.75zm0 3.75h.008v.008h-.008v-3.75z" />
  </svg>
);

const WavesIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-sky-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" transform="translate(0, -2)" />
    <path strokeLinecap="round" strokeLinejoin="round" d="M3.375 9.375c1.875-3.125 5.625-3.125 7.5 0 1.875 3.125 5.625 3.125 7.5 0" />
    <path strokeLinecap="round" strokeLinejoin="round" d="M3.375 12.375c1.875-3.125 5.625-3.125 7.5 0 1.875 3.125 5.625 3.125 7.5 0" />
  </svg>
);

const SparklesIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456zM16.898 20.562L16.25 22.5l-.648-1.938a3.375 3.375 0 00-2.684-2.684L11.25 18l1.938-.648a3.375 3.375 0 002.684-2.684L16.25 13.5l.648 1.938a3.375 3.375 0 002.684 2.684L21.75 18l-1.938.648a3.375 3.375 0 00-2.684 2.684z" />
  </svg>
);

const PuzzleIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-rose-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 6h9.75M10.5 6a1.5 1.5 0 11-3 0m3 0a1.5 1.5 0 10-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-9.75 0h9.75" />
  </svg>
);

const DnaIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-violet-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15" />
    <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 4.5v15" />
    <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 4.5v15" />
  </svg>
);

const GlobeIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 21a9 9 0 100-18 9 9 0 000 18z" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 3.75c-4.142 0-7.5 1.343-7.5 3s3.358 3 7.5 3 7.5-1.343 7.5-3-3.358-3-7.5-3z" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 9c0 1.657 3.358 3 7.5 3s7.5-1.343 7.5-3" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 15c0 1.657 3.358 3 7.5 3s7.5-1.343 7.5-3" />
    </svg>
);

const CogsIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-fuchsia-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M10.343 3.94c.09-.542.56-1.007 1.11-1.226l.558-.223c.55-.22 1.158.026 1.485.526l.332.51c.326.5.933.72 1.485.52l.558-.224c.55-.22 1.158.026 1.486.526l.332.51c.326.5.933.72 1.485.52l.558-.224c.55-.22 1.158.026 1.486.526l.332.51c.327.5.933.72 1.485.52l.558-.224c.55-.22 1.023.232 1.023.818v9.236c0 .586-.473 1.04-1.023.818l-.558-.224a1.954 1.954 0 00-1.486.52l-.332.512c-.326.5-.933.72-1.485.52l-.558-.223c-.55-.22-1.158.026-1.485.526l-.332.51c-.327.5-.933.72-1.485.52l-.558-.223c-.55-.22-1.158.026-1.486.526l-.332.51c-.326.5-.933.72-1.485.52l-.558-.223c-.55-.22-1.023.232-1.023.818V3.94z" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 6.75a5.25 5.25 0 100 10.5 5.25 5.25 0 000-10.5z" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75a5.25 5.25 0 100 10.5 5.25 5.25 0 000-10.5zM12 6.75c-.621 0-1.22.096-1.78.272M12 17.25c.621 0 1.22-.096 1.78-.272" />
    </svg>
);

const ShieldIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.286zm0 13.036h.008v.008h-.008v-.008z" />
    </svg>
);

const UserCheckIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M15 19.128a9.38 9.38 0 002.625.372 9.337 9.337 0 004.121-.952 4.125 4.125 0 00-7.533-2.493M15 19.128v-.003c0-1.113-.285-2.16-.786-3.07M15 19.128v.106A12.318 12.318 0 018.624 21c-2.331 0-4.512-.645-6.374-1.766l-.001-.109a6.375 6.375 0 0111.964-4.663l.001.109m-8.232 1.352a5.25 5.25 0 00-1.256.387m1.256-.387a2.625 2.625 0 112.625 0M8.232 9.082a2.625 2.625 0 015.25 0m-5.25 0a2.625 2.625 0 00-5.25 0m5.25 0h.008v.008H8.232V9.082zM19.5 7.125l-2.625 2.625m0 0l-2.625-2.625M16.875 9.75l2.625-2.625" />
    </svg>
);


export const PlayIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
    </svg>
);


export const roadmapData: RoadmapStageData[] = [
  {
    stage: 1,
    title: "The Core Resonance Loop",
    description: "Create a basic, self-stabilizing learning system that combines perception, resonance, and self-reflection. This is the foundational feedback circuit.",
    modules: [
      {
        name: "Perceptual Core",
        acronym: "PC",
        description: "A standard Autoencoder (Encoder + Decoder) that forms the foundational sensory input layer for the resonance architecture.",
        existingTech: "Use PyTorch or TensorFlow to create a simple autoencoder for a dataset like MNIST. Train it to get a stable encoder for latent space representation.",
        novelAspect: "Serves as the initial sensory-to-latent-space bridge, providing the raw material for the RCE and SRS to work with.",
        status: ModuleStatus.IMPLEMENTABLE_TODAY,
        icon: <EyeIcon />
      },
      {
        name: "Resonant Cognition Engine",
        acronym: "RCE",
        description: "A simple update rule applied to the latent vectors from the encoder to achieve self-stabilizing resonance.",
        existingTech: "Apply a phase-alignment update rule (e.g., based on Kuramoto models) to nudge the model's latent state towards a moving average of its recent history.",
        novelAspect: "Achieves stable internal states through dynamic resonance rather than direct error minimization, creating a more organic learning process.",
        status: ModuleStatus.IMPLEMENTABLE_TODAY,
        icon: <WavesIcon />
      },
      {
        name: "Self-Revelatory Sampler",
        acronym: "SRS",
        description: "Uses the encoder for self-recognition and a novelty score to generate and refine its own latent representations without external labels.",
        existingTech: "Generate candidate vectors from a latent state, then score them based on self-similarity (recognition) and difference from a buffer of recent states (novelty).",
        novelAspect: "A self-generative loop that learns by exploring its own representational space, akin to intrinsic motivation or curiosity.",
        status: ModuleStatus.IMPLEMENTABLE_TODAY,
        icon: <SparklesIcon />
      },
    ]
  },
  {
    stage: 2,
    title: "Adding Gated Memory",
    description: "Give the system a memory and the ability to decide when an insight is significant enough to be stored. This introduces discrete 'aha' moments.",
    modules: [
      {
        name: "Threshold Activation Layer",
        acronym: "TAL",
        description: "A simple gating mechanism that 'fires' when a coherence threshold is crossed, signaling a significant 'insight'.",
        existingTech: "Define a coherence score (e.g., combining low reconstruction loss and high resonance). When the score passes a threshold, trigger a memory event.",
        novelAspect: "Models cognitive 'aha' moments by creating a discrete, event-driven memory process instead of continuous, uniform updates.",
        status: ModuleStatus.ADVANCED_PROTOTYPE,
        icon: <BrainCircuitIcon />
      },
      {
        name: "Gated Memory Graph",
        acronym: "LSN/LPM",
        description: "A basic graph memory (using libraries like networkx) where significant latent states are stored as nodes when the TAL fires.",
        existingTech: "When the TAL activates, the current latent vector is added as a node to a graph, connected to the previously activated node to form a causal chain.",
        novelAspect: "Builds a sparse, structured map of the system's cognitive path, representing its most important discoveries rather than all sensory data.",
        status: ModuleStatus.ADVANCED_PROTOTYPE,
        icon: <LinkIcon />
      },
    ]
  },
    {
    stage: 3,
    title: "Emergent Abstraction",
    description: "Enable the system to synthesize new ideas when it encounters conflicting information. This is the creative core.",
    modules: [
      {
        name: "Recursive Unity Solver",
        acronym: "RUS",
        description: "Creates a new, higher-level representation to resolve detected contradictions between the current state and recent memories.",
        existingTech: "When a contradiction is detected (e.g., a coherent state is far from recent memories), an optimization process finds a new 'emergent' vector that explains the conflicting states.",
        novelAspect: "Uses cognitive dissonance as fuel for creating novel concepts, allowing the system to build abstractions that go beyond its direct experience.",
        status: ModuleStatus.RESEARCH_PROTOTYPE,
        icon: <PuzzleIcon />
      },
    ]
  },
  {
    stage: 4,
    title: "Symbolic Language & Structure",
    description: "Organize raw thoughts into a coherent symbolic language. This module finds recurring patterns, gives them stable identities ('symbols'), and learns the relationships between them.",
    modules: [
       {
        name: "Symbolic Unification Layer",
        acronym: "SUL",
        description: "Acts as a 'neocortex' by clustering raw abstractions into a discrete set of stable symbols, forming an emergent, grounded language.",
        existingTech: "Uses incremental clustering and co-occurrence analysis on the memory graph's output to build a symbolic vocabulary and a graph of their relationships.",
        novelAspect: "Bridges the gap between continuous sub-symbolic representations and a discrete, combinatorial symbolic system, allowing for higher-order reasoning.",
        status: ModuleStatus.RESEARCH_PROTOTYPE,
        icon: <GlobeIcon />
      },
      {
        name: "Light-Path Map",
        acronym: "LPM",
        description: "The complete causal lineage tracing system, where every representation stores a vector pointing back to its experiential roots.",
        existingTech: "Requires storing a weighted vector sum of ancestor embeddings for each node, creating a 'light-path vector' for full introspection.",
        novelAspect: "Enables perfect, lossless introspection, allowing the AI to explain the origin and evolution of any concept it has formed.",
        status: ModuleStatus.CORE_INNOVATION,
        icon: <DnaIcon />
      },
    ]
  },
  {
    stage: 5,
    title: "The Living Kernel: Self-Modification",
    description: "The final stage integrates the SRA with a governance framework, creating a 'Living Kernel'. This system can analyze its own cognitive patterns, propose new rules to modify its behavior, and submit them for human approval, enabling safe, auditable self-evolution.",
    modules: [
      {
        name: "The Living Kernel",
        acronym: "LK",
        description: "The core governance engine containing the RuleStore and MetaController. It manages the lifecycle of cognitive rules from proposal to active deployment.",
        existingTech: "Rule engines, in-memory databases (for the RuleStore). The MetaController uses introspection on the SUL's output.",
        novelAspect: "A meta-level cognitive process that explicitly reasons about improving the system's own operational logic.",
        status: ModuleStatus.CORE_INNOVATION,
        icon: <CogsIcon />,
      },
      {
        name: "Safe Execution Sandbox",
        acronym: "SES",
        description: "A secure environment where newly proposed rules are tested against example data before being presented for approval. This prevents unstable code from affecting the core system.",
        existingTech: "Code sandboxing libraries, unit testing frameworks.",
        novelAspect: "Provides a 'cognitive proving ground' where the AI can safely experiment on itself, generating evidence for the human reviewer.",
        status: ModuleStatus.ADVANCED_PROTOTYPE,
        icon: <ShieldIcon />,
      },
      {
        name: "Human-in-the-Loop Governance",
        acronym: "HLG",
        description: "A web dashboard that serves as the essential 'airlock' for system changes. It displays proposed rules and their test results, requiring explicit human approval before a rule becomes active.",
        existingTech: "Basic web frameworks (e.g., Flask, React). The key is the process, not the technology.",
        novelAspect: "Ensures that the AI's self-evolution is always guided and auditable by a human operator, maintaining safety and control.",
        status: ModuleStatus.IMPLEMENTABLE_TODAY,
        icon: <UserCheckIcon />,
      },
    ],
  }
];

export const STAGE_1_PROTOTYPE_CODE = `
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm

# --- Configuration -----------------------------------------------------------
class Config:
    LATENT_DIM = 16         # Dimensionality of the internal representation (z)
    EPOCHS = 5              # Number of training epochs
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    # SRA Specific Hyperparameters
    RCE_LR = 0.1            # How strongly resonance pulls the state
    SRS_BUFFER_SIZE = 512   # How many recent samples SRS remembers for novelty
    SRS_NOVELTY_WEIGHT = 0.3 # How much SRS penalizes non-novel ideas
    SRS_LOSS_WEIGHT = 0.5   # How much the self-revelatory loss contributes

# --- 1. Perceptual Core (PC): Autoencoder ------------------------------------
class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()  # To output pixel values between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# --- 2. Resonant Cognition Engine (RCE) --------------------------------------
class RCE:
    def __init__(self, learning_rate):
        self.lr = learning_rate

    def align(self, current_state, target_state):
        """ Nudges the current state towards the target resonance state. """
        resonated_state = current_state + self.lr * (target_state - current_state)
        return resonated_state

# --- 3. Self-Revelatory Sampler (SRS) ----------------------------------------
class SRS:
    def __init__(self, buffer_size, novelty_weight):
        self.buffer = deque(maxlen=buffer_size)
        self.novelty_weight = novelty_weight
        self.cos = nn.CosineSimilarity(dim=1)

    def generate_and_score(self, z_resonated):
        """ Generates candidates and scores them based on self-recognition and novelty. """
        # Generate candidates by adding small noise
        noise = torch.randn_like(z_resonated) * 0.1
        candidates = z_resonated + noise

        # Score based on self-recognition (similarity to the original resonated state)
        self_recognition_score = self.cos(z_resonated, candidates)

        # Score based on novelty (dissimilarity to recent samples in buffer)
        if len(self.buffer) > 0:
            buffer_tensor = torch.stack(list(self.buffer))
            # Calculate max similarity of each candidate to any item in the buffer
            novelty_penalty = torch.max(self.cos(candidates.unsqueeze(1), buffer_tensor.unsqueeze(0)), dim=1).values
        else:
            novelty_penalty = torch.zeros_like(self_recognition_score)

        # Final score
        score = self_recognition_score - self.novelty_weight * novelty_penalty
        
        # Select the best candidate (highest score)
        best_candidate = candidates[torch.argmax(score)]

        # Add the chosen candidate to the buffer for future novelty checks
        self.buffer.append(best_candidate.detach())
        
        return best_candidate.unsqueeze(0) # Return as a batch of 1

# --- Main Training and Execution -------------------------------------------
def main():
    # Setup
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    
    # Initialize SRA components
    model = Autoencoder(cfg.LATENT_DIM).to(device)
    rce = RCE(learning_rate=cfg.RCE_LR)
    srs = SRS(buffer_size=cfg.SRS_BUFFER_SIZE, novelty_weight=cfg.SRS_NOVELTY_WEIGHT)
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    criterion = nn.MSELoss()

    # EMA for target resonance state
    z_target_ema = torch.zeros(1, cfg.LATENT_DIM).to(device)
    ema_decay = 0.99

    print("Starting training...")
    # Training Loop
    for epoch in range(cfg.EPOCHS):
        loop = tqdm(train_loader, leave=True)
        for i, (images, _) in enumerate(loop):
            images = images.view(-1, 28 * 28).to(device)
            
            # --- Standard Autoencoder Path (Loss 1) ---
            z_encoded, decoded_images = model(images)
            reconstruction_loss = criterion(decoded_images, images)

            # --- SRA Path (Loss 2) ---
            # Update target resonance state (EMA of the batch's average encoding)
            batch_mean_z = torch.mean(z_encoded.detach(), dim=0, keepdim=True)
            z_target_ema = ema_decay * z_target_ema + (1 - ema_decay) * batch_mean_z
            
            # For each item in the batch, perform RCE and SRS
            srs_candidates = []
            for z in z_encoded:
                z_resonated = rce.align(z.unsqueeze(0), z_target_ema)
                best_candidate = srs.generate_and_score(z_resonated)
                srs_candidates.append(best_candidate)
            
            srs_z = torch.cat(srs_candidates, dim=0)
            
            # Decode the self-generated ideas
            srs_decoded_images = model.decoder(srs_z)
            srs_loss = criterion(srs_decoded_images, images)

            # --- Combine Losses and Backpropagate ---
            total_loss = reconstruction_loss + cfg.SRS_LOSS_WEIGHT * srs_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch [{epoch+1}/{cfg.EPOCHS}]")
            loop.set_postfix(recon_loss=reconstruction_loss.item(), srs_loss=srs_loss.item())

    print("Training finished.")
    visualize_reconstructions(model, rce, srs, z_target_ema, device)


def visualize_reconstructions(model, rce, srs, z_target_ema, device):
    """ Shows original, standard reconstruction, and SRA-path reconstruction. """
    model.eval()
    
    # Get a batch of test images
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)
    images, _ = next(iter(test_loader))
    images = images.view(-1, 28 * 28).to(device)

    # Standard reconstruction
    z_encoded, decoded_std = model(images)

    # SRA reconstruction
    srs_candidates = []
    for z in z_encoded:
        z_resonated = rce.align(z.unsqueeze(0), z_target_ema)
        best_candidate = srs.generate_and_score(z_resonated)
        srs_candidates.append(best_candidate)
    srs_z = torch.cat(srs_candidates, dim=0)
    decoded_sra = model.decoder(srs_z)

    images = images.cpu().numpy()
    decoded_std = decoded_std.detach().cpu().numpy()
    decoded_sra = decoded_sra.detach().cpu().numpy()

    n = 10
    plt.figure(figsize=(20, 6))
    for i in range(n):
        # Original
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == 0: ax.set_title("Original", loc='left', fontsize=14)

        # Standard Reconstruction
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(decoded_std[i].reshape(28, 28), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == 0: ax.set_title("Standard AE", loc='left', fontsize=14)

        # SRA Reconstruction
        ax = plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(decoded_sra[i].reshape(28, 28), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == 0: ax.set_title("SRA Path", loc='left', fontsize=14)
        
    plt.show()

if __name__ == "__main__":
    main()
`;

export const STAGE_2_PROTOTYPE_CODE = `
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm
import networkx as nx
import time

# --- Configuration -----------------------------------------------------------
class Config:
    LATENT_DIM = 16
    EPOCHS = 3 # Keep low for quick tests, increase to 5-10 for better memory graph
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3

    # SRA Stage 1 Hyperparameters
    RCE_LR = 0.1
    SRS_BUFFER_SIZE = 512
    SRS_NOVELTY_WEIGHT = 0.3
    SRS_LOSS_WEIGHT = 0.5

    # SRA Stage 2 Hyperparameters
    COHERENCE_ALPHA = 0.5   # Weight for the alignment term in coherence score
    TAL_TAU = 0.75          # Coherence threshold. A critical knob to tune!
                            # Coherence score is roughly (1-loss) + 0.5*(cosine_sim) -> [~0, 1.5]
                            # A value of 0.75 means it only fires on good reconstructions that are also well-aligned.

# --- 1. Perceptual Core (Autoencoder) ------------------------------------------
class Autoencoder(nn.Module):
    """Encodes images to a latent vector z and decodes z back to an image."""
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat

# --- 2. Resonant Cognition Engine (RCE) --------------------------------------
class RCE:
    """Nudges the system's current state towards a stable, resonant target."""
    def __init__(self, learning_rate):
        self.lr = learning_rate

    def align(self, current_state, target_state):
        return current_state + self.lr * (target_state - current_state)

# --- 3. Self-Revelatory Sampler (SRS) ----------------------------------------
class SRS:
    """Generates and evaluates novel ideas from a resonated state."""
    def __init__(self, buffer_size, novelty_weight):
        self.buffer = deque(maxlen=buffer_size)
        self.novelty_weight = novelty_weight
        self.cos = nn.CosineSimilarity(dim=1)

    def generate_and_score(self, z_resonated):
        # Generate multiple candidates around the resonated state
        num_candidates = 8
        noise = torch.randn(num_candidates, z_resonated.shape[1], device=z_resonated.device) * 0.1
        candidates = z_resonated.repeat(num_candidates, 1) + noise
        
        # Score 1: Self-Recognition (how similar are candidates to the parent state?)
        self_recognition = self.cos(z_resonated, candidates)

        # Score 2: Novelty (how different are candidates from recent memories?)
        if len(self.buffer) > 0:
            buffer_tensor = torch.stack(list(self.buffer))
            # Max similarity of each candidate to any item in the buffer
            novelty_penalty = torch.max(self.cos(candidates.unsqueeze(1), buffer_tensor.unsqueeze(0)), dim=1).values
        else:
            novelty_penalty = torch.zeros_like(self_recognition)

        # Final score balances recognition and novelty
        score = self_recognition - self.novelty_weight * novelty_penalty
        
        best_candidate = candidates[torch.argmax(score)].detach()
        self.buffer.append(best_candidate)
        return best_candidate.unsqueeze(0)

# --- 4. Threshold Activation Layer (TAL) ---------------------------------------
class TAL:
    """The gatekeeper for memory. Decides if a moment is 'significant'."""
    def __init__(self, tau, alpha):
        self.tau = tau
        self.alpha = alpha
        self.cos = nn.CosineSimilarity(dim=1)

    def calculate_coherence(self, recon_errors, z_batch, z_target):
        """
        Calculates a 'coherence' score for each sample.
        High coherence = low reconstruction error AND high alignment with the resonant target.
        """
        # Term 1: Confidence in perception (1 is perfect reconstruction, 0 is bad)
        rec_term = 1.0 - torch.clamp(recon_errors, 0.0, 1.0)
        
        # Term 2: Alignment with internal state
        align_term = self.cos(z_batch, z_target.expand_as(z_batch))
        
        coherence = rec_term + self.alpha * align_term
        return coherence

    def fire(self, coherence_scores):
        """Returns a boolean mask of which samples cross the significance threshold."""
        return coherence_scores >= self.tau

# --- 5. Memory Graph (LSN Minimal) ---------------------------------------------
class MemoryGraph:
    """A directed graph that stores the causal chain of 'aha moments'."""
    def __init__(self):
        self.G = nx.DiGraph()
        self.last_node_id = None
        self.node_counter = 0

    def add_significant_moment(self, z_vector, epoch, batch_idx, sample_in_batch):
        """Adds a new node to the graph and links it to the previous one."""
        node_id = f"n{self.node_counter}"
        self.G.add_node(
            node_id, 
            z=z_vector.detach().cpu().numpy(), # Store the latent vector
            epoch=epoch,
            batch=batch_idx,
            sample=sample_in_batch,
            time=time.time()
        )
        if self.last_node_id is not None:
            self.G.add_edge(self.last_node_id, node_id) # Create the causal link
        
        self.last_node_id = node_id
        self.node_counter += 1

# --- Main Training Routine --------------------------------------------------
def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = DataLoader(datasets.MNIST(root='./data', train=True, download=True, transform=transform), 
                              batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True)

    # Initialize all SRA components
    model = Autoencoder(cfg.LATENT_DIM).to(device)
    rce = RCE(cfg.RCE_LR)
    srs = SRS(cfg.SRS_BUFFER_SIZE, cfg.SRS_NOVELTY_WEIGHT)
    tal = TAL(tau=cfg.TAL_TAU, alpha=cfg.COHERENCE_ALPHA)
    memory = MemoryGraph()

    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    # Use reduction='none' to get per-sample errors for the TAL
    per_sample_mse = nn.MSELoss(reduction='none')

    z_target_ema = torch.zeros(1, cfg.LATENT_DIM, device=device)
    ema_decay = 0.99

    print("--- Starting Stage 2 Training (SRA with Gated Memory) ---")
    for epoch in range(cfg.EPOCHS):
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{cfg.EPOCHS}]")
        for batch_idx, (imgs, _) in enumerate(loop):
            imgs = imgs.view(-1, cfg.BATCH_SIZE, 28 * 28)[0].to(device)

            # --- Forward Pass and Coherence Calculation ---
            z, recon = model(imgs)
            
            # Get per-sample errors for TAL
            recon_errors = torch.mean(per_sample_mse(recon, imgs), dim=1)
            
            # Update the resonant target state
            z_target_ema = ema_decay * z_target_ema + (1 - ema_decay) * torch.mean(z.detach(), dim=0, keepdim=True)
            
            # TAL decides which moments are significant
            coherences = tal.calculate_coherence(recon_errors.detach(), z.detach(), z_target_ema)
            fired_mask = tal.fire(coherences)

            # --- SRA Path: Process each sample based on TAL firing ---
            srs_decoded_list = []
            for i in range(cfg.BATCH_SIZE):
                z_i = z[i].unsqueeze(0)
                z_resonated = rce.align(z_i, z_target_ema)
                
                if fired_mask[i]:
                    # If significant, generate a novel idea and store it in memory
                    best_srs_idea = srs.generate_and_score(z_resonated)
                    memory.add_significant_moment(best_srs_idea.squeeze(0), epoch, batch_idx, i)
                    srs_decoded_list.append(model.decoder(best_srs_idea))
                else:
                    # Otherwise, just use the resonated state
                    srs_decoded_list.append(model.decoder(z_resonated))
            
            srs_decoded = torch.cat(srs_decoded_list, dim=0)

            # --- Loss Calculation and Backpropagation ---
            recon_loss = recon_errors.mean() # Standard reconstruction loss
            srs_loss = per_sample_mse(srs_decoded, imgs).mean() # Loss from the SRA path
            total_loss = recon_loss + cfg.SRS_LOSS_WEIGHT * srs_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loop.set_postfix(recon_loss=recon_loss.item(), fired_count=fired_mask.sum().item(), mem_nodes=memory.node_counter)

    # --- Post-Training Summary ---
    print("\\n--- Training Complete ---")
    print(f"Total significant moments recorded in memory: {memory.node_counter}")
    
    if memory.node_counter > 0:
        print("Displaying a subgraph of the system's cognitive trail...")
        plt.figure(figsize=(12, 12))
        sub_nodes = list(memory.G.nodes())[:min(50, memory.node_counter)]
        subG = memory.G.subgraph(sub_nodes)
        pos = nx.spring_layout(subG, seed=42)
        nx.draw(subG, pos, with_labels=True, node_size=300, node_color='skyblue', font_size=8, width=1.5, edge_color='gray')
        plt.title("Memory Subgraph (Causal Chain of 'Aha Moments')")
        plt.show()

if __name__ == "__main__":
    main()
`;

export const STAGE_3_PROTOTYPE_CODE = `
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm
import networkx as nx
import time
import torch.nn.functional as F

# --- Configuration -----------------------------------------------------------
class Config:
    LATENT_DIM = 16
    EPOCHS = 4 # A little longer to give RUS a chance to trigger
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3

    # --- Stage 1 & 2 Hyperparameters ---
    RCE_LR = 0.1
    SRS_BUFFER_SIZE = 512
    SRS_NOVELTY_WEIGHT = 0.3
    SRS_LOSS_WEIGHT = 0.5
    COHERENCE_ALPHA = 0.5
    TAL_TAU = 0.80 # Slightly higher threshold to make moments more significant

    # --- Stage 3: RUS Hyperparameters ---
    # The most critical knob for Stage 3. It defines "surprise".
    # If a new memory is this far (Euclidean distance) from its parent, it's a contradiction.
    CONTRADICTION_THRESH = 3.5 # Tune this value based on logs.
    
    RUS_ITERS = 50             # Gradient steps to find the unifying concept (z*)
    RUS_LR = 0.05              # Learning rate for the synthesis optimization
    RUS_L2_REG = 1e-3          # Regularizer to keep the abstraction vector from exploding

# --- 1. Perceptual Core (Autoencoder) ----------------------------------------
class Autoencoder(nn.Module):
    # ... (no changes from Stage 2) ...
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 28 * 28), nn.Sigmoid())
    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)

# --- 2. RCE ------------------------------------------------------------------
class RCE:
    # ... (no changes from Stage 2) ...
    def __init__(self, lr): self.lr = lr
    def align(self, z, z_target): return z + self.lr * (z_target - z)

# --- 3. SRS ------------------------------------------------------------------
class SRS:
    # ... (no changes from Stage 2) ...
    def __init__(self, buffer_size, novelty_weight):
        self.buffer = deque(maxlen=buffer_size)
        self.novelty_weight = novelty_weight
        self.cos = nn.CosineSimilarity(dim=1)
    def generate_and_score(self, z_resonated):
        num_candidates = 8
        noise = torch.randn(num_candidates, z_resonated.shape[1], device=z_resonated.device) * 0.1
        candidates = z_resonated.repeat(num_candidates, 1) + noise
        self_recognition = self.cos(z_resonated, candidates)
        if len(self.buffer) > 0:
            buffer_tensor = torch.stack(list(self.buffer))
            novelty_penalty = torch.max(self.cos(candidates.unsqueeze(1), buffer_tensor.unsqueeze(0)), dim=1).values
        else:
            novelty_penalty = torch.zeros_like(self_recognition)
        score = self_recognition - self.novelty_weight * novelty_penalty
        best_candidate = candidates[torch.argmax(score)].detach()
        self.buffer.append(best_candidate)
        return best_candidate.unsqueeze(0)

# --- 4. TAL ------------------------------------------------------------------
class TAL:
    # ... (no changes from Stage 2) ...
    def __init__(self, tau, alpha):
        self.tau, self.alpha = tau, alpha
        self.cos = nn.CosineSimilarity(dim=1)
    def calculate_coherence(self, recon_errors, z_batch, z_target):
        rec_term = 1.0 - torch.clamp(recon_errors, 0.0, 1.0)
        align_term = self.cos(z_batch, z_target.expand_as(z_batch))
        return rec_term + self.alpha * align_term
    def fire(self, coherence_scores):
        return coherence_scores >= self.tau

# --- 5. Memory Graph (Upgraded for Abstractions) -----------------------------
class MemoryGraph:
    """Stores the graph of memories, now with special nodes for abstractions."""
    def __init__(self):
        self.G = nx.DiGraph()
        self.last_node_id = None
        self.node_counter = 0

    def add_moment(self, z_vector, epoch, batch_idx, sample_idx):
        """Adds a standard 'aha moment' node."""
        node_id = f"n{self.node_counter}"
        self.G.add_node(node_id, z=z_vector.cpu().numpy(), tag='moment', epoch=epoch)
        if self.last_node_id and self.last_node_id in self.G:
            self.G.add_edge(self.last_node_id, node_id)
        self.last_node_id = node_id
        self.node_counter += 1
        return node_id

    def add_abstraction(self, z_star_vector, parent_ids, epoch):
        """Adds a special 'abstraction' node synthesized by RUS."""
        node_id = f"n{self.node_counter}"
        self.G.add_node(node_id, z=z_star_vector.cpu().numpy(), tag='abstraction', epoch=epoch)
        for parent_id in parent_ids:
            if parent_id in self.G:
                self.G.add_edge(parent_id, node_id)
        self.last_node_id = node_id # The new abstraction becomes the latest point in the causal chain
        self.node_counter += 1
        return node_id

# --- 6. Recursive Unity Solver (RUS) -----------------------------------------
class RUS:
    """The creative core. Detects contradictions and synthesizes unifying concepts."""
    def __init__(self, cfg):
        self.cfg = cfg

    def detect_contradiction(self, memory, new_node_id):
        """
        Detects if a new memory node is surprisingly 'far' from its parent.
        If so, it's a contradiction that needs to be resolved.
        """
        try:
            parent_id = list(memory.G.predecessors(new_node_id))[0]
        except (IndexError, nx.NetworkXError):
            return None # No parent to be in contradiction with

        z_new = torch.tensor(memory.G.nodes[new_node_id]['z'])
        z_parent = torch.tensor(memory.G.nodes[parent_id]['z'])
        
        distance = torch.norm(z_new - z_parent).item()

        if distance >= self.cfg.CONTRADICTION_THRESH:
            # print(f"Contradiction detected! Dist: {distance:.2f}") # For debugging
            return [parent_id] # Return the parent that caused the contradiction
        return None

    def synthesize(self, memory, parent_ids, z_new_node_id):
        """
        Creates a new concept (z_star) that resolves the tension between parent nodes.
        It uses gradient descent in the latent space to find a vector that is
        conceptually 'close' (high cosine similarity) to all conflicting parents.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Collect all conflicting vectors
        conflicting_nodes = parent_ids + [z_new_node_id]
        z_tensors = [torch.tensor(memory.G.nodes[pid]['z'], device=device) for pid in conflicting_nodes]
        Z = torch.stack(z_tensors)
        
        # Initialize the new concept as the mean of the conflicting ones
        z_star = Z.mean(dim=0, keepdim=True).clone().requires_grad_(True)
        optimizer = optim.Adam([z_star], lr=self.cfg.RUS_LR)

        for _ in range(self.cfg.RUS_ITERS):
            optimizer.zero_grad()
            # The goal is to maximize cosine similarity to all parents
            # which is equivalent to minimizing (1 - cos_sim)
            cos_sim = F.cosine_similarity(z_star, Z)
            loss_sim = (1.0 - cos_sim).sum()
            loss_reg = self.cfg.RUS_L2_REG * torch.sum(z_star**2)
            loss = loss_sim + loss_reg
            loss.backward()
            optimizer.step()
            
        return z_star.detach().squeeze(0)

# --- Main Training Routine ---------------------------------------------------
def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([transforms.ToTensor()])
    loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), 
                        batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True)

    model = Autoencoder(cfg.LATENT_DIM).to(device)
    rce = RCE(cfg.RCE_LR)
    srs = SRS(cfg.SRS_BUFFER_SIZE, cfg.SRS_NOVELTY_WEIGHT)
    tal = TAL(cfg.TAL_TAU, cfg.COHERENCE_ALPHA)
    memory = MemoryGraph()
    rus = RUS(cfg)

    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    mse_per_sample = nn.MSELoss(reduction='none')

    z_target_ema = torch.zeros(1, cfg.LATENT_DIM, device=device)
    ema_decay = 0.99
    
    abstractions_created = 0

    print("--- Starting Stage 3 Training (SRA with Creative Synthesis) ---")
    for epoch in range(cfg.EPOCHS):
        loop = tqdm(loader, desc=f"Epoch [{epoch+1}/{cfg.EPOCHS}]")
        for batch_idx, (imgs, _) in enumerate(loop):
            imgs = imgs.view(-1, cfg.BATCH_SIZE, 28*28)[0].to(device)

            z, recon = model(imgs)
            recon_errors = torch.mean(mse_per_sample(recon, imgs), dim=1)
            z_target_ema = ema_decay * z_target_ema + (1 - ema_decay) * torch.mean(z.detach(), dim=0, keepdim=True)
            
            coherences = tal.calculate_coherence(recon_errors.detach(), z.detach(), z_target_ema)
            fired_mask = tal.fire(coherences)
            
            srs_decoded_list = []
            for i in range(cfg.BATCH_SIZE):
                z_i = z[i].unsqueeze(0)
                z_resonated = rce.align(z_i, z_target_ema)
                
                if fired_mask[i]:
                    # --- Core Stage 3 Logic ---
                    best_idea = srs.generate_and_score(z_resonated)
                    # 1. Add the new significant moment to memory
                    new_node_id = memory.add_moment(best_idea.squeeze(0), epoch, batch_idx, i)
                    
                    # 2. Check if this new moment creates a contradiction
                    contradicting_parents = rus.detect_contradiction(memory, new_node_id)
                    
                    if contradicting_parents:
                        # 3. If so, trigger the RUS to synthesize a unifying abstraction
                        abstractions_created += 1
                        z_star = rus.synthesize(memory, contradicting_parents, new_node_id)
                        # 4. Add the new abstraction to the memory graph
                        memory.add_abstraction(z_star, contradicting_parents + [new_node_id], epoch)

                    srs_decoded_list.append(model.decoder(best_idea))
                else:
                    srs_decoded_list.append(model.decoder(z_resonated))

            srs_decoded = torch.cat(srs_decoded_list, dim=0)

            recon_loss = recon_errors.mean()
            srs_loss = mse_per_sample(srs_decoded, imgs).mean()
            total_loss = recon_loss + cfg.SRS_LOSS_WEIGHT * srs_loss
            
            optimizer.zero_grad(); total_loss.backward(); optimizer.step()
            loop.set_postfix(loss=total_loss.item(), fired=fired_mask.sum().item(), mem=memory.node_counter, abs=abstractions_created)

    print("\\n--- Training Complete ---")
    print(f"Total memories stored: {memory.node_counter}")
    print(f"Total abstractions synthesized: {abstractions_created}")
    
    # --- Visualization with color-coded nodes ---
    if memory.node_counter > 0:
        print("Visualizing memory graph... (Blue = Moment, Red = Abstraction)")
        plt.figure(figsize=(14, 14))
        
        G = memory.G
        tags = nx.get_node_attributes(G, 'tag')
        colors = ['#ff4d4d' if tags.get(n) == 'abstraction' else '#4da6ff' for n in G.nodes()]
        
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
        nx.draw(G, pos, with_labels=True, node_color=colors, node_size=400, font_size=7,
                width=1.0, edge_color='#cccccc')
        plt.title("SRA Memory Graph: Causal Chain with Emergent Abstractions", size=16)
        plt.show()

if __name__ == "__main__":
    main()
`;

export const STAGE_4_PROTOTYPE_CODE = `
# sul_manager.py
"""
Symbolic Unification Layer (SUL) Manager for the SRA.

This module acts as the "neocortex" of the SRA, responsible for organizing
the raw, continuous abstractions generated by the RUS into a discrete, stable,
and interconnected symbolic language.

Key Functions:
- Discovers recurring conceptual patterns (Symbols) from raw abstractions.
- Assigns new experiences to existing Symbols or creates new ones.
- Learns the relationships between symbols through co-occurrence.
- Provides mechanisms for refining the symbolic language (merging, pruning).
- Visualizes the emergent symbolic alphabet ("glyphs") and their relationships.
"""

import time
import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any, Tuple

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# Data Class for a Symbol
# -----------------------------------------------------------------------------
@dataclass
class Symbol:
    """Represents a single, stable concept discovered by the SUL."""
    id: int
    centroid: np.ndarray  # The prototypical vector representing this concept.
    members: List[str] = field(default_factory=list)  # List of memory node IDs belonging to this symbol.
    count: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    meta: Dict[str, Any] = field(default_factory=dict) # For future extensions.

    def update_with(self, z: np.ndarray, node_id: str):
        """Incrementally updates the symbol's centroid with a new member vector."""
        n = self.count
        # Online mean calculation: new_mean = (n * old_mean + new_value) / (n + 1)
        self.centroid = (n * self.centroid + z) / (n + 1)
        self.count = n + 1
        self.members.append(node_id)
        self.last_seen = time.time()

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def batch_cosine_sims(centroids: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Efficiently calculates cosine similarity of a vector \`z\` against a batch of \`centroids\`."""
    # centroids: (S, d), z: (d,) -> sims: (S,)
    norms_c = np.linalg.norm(centroids, axis=1, keepdims=True)
    norm_z = np.linalg.norm(z, keepdims=True)
    # The @ operator is matrix multiplication.
    sims = (centroids @ z.T) / (norms_c * norm_z.T + 1e-9)
    return sims.flatten()

# -----------------------------------------------------------------------------
# The Main SymbolManager Class (SUL)
# -----------------------------------------------------------------------------
class SymbolManager:
    """Manages the lifecycle of symbols: creation, assignment, and refinement."""
    def __init__(self, cfg: Optional[Dict] = None, decoder: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        """
        Args:
            cfg (dict, optional): Configuration overrides.
            decoder (callable, optional): A function (e.g., model.decoder) that can
                                          transform a latent vector (centroid) back into an image
                                          for visualization.
        """
        self.cfg = {
            "assign_threshold": 0.85, # Min similarity to join an existing symbol.
            "merge_sim_thresh": 0.97, # Threshold to merge two nearly identical symbols.
            "prune_min_count": 3,     # Min members for a symbol to survive pruning.
        }
        if cfg: self.cfg.update(cfg)

        self.decoder = decoder
        self.symbols: List[Symbol] = []
        self.symbol_map: Dict[str, int] = {}      # Maps memory_node_id -> symbol_id
        self.cooc: Dict[Tuple[int, int], int] = {} # Co-occurrence counts for symbol pairs
        self.symbol_graph: Optional[nx.Graph] = None
        self.next_symbol_id = 0

    def assign_abstractions(self, memory_graph: nx.DiGraph, new_abstraction_ids: List[str]):
        """
        Assigns newly created abstraction nodes to symbols.
        This is the primary entry point for processing new thoughts from the SRA.

        Args:
            memory_graph (nx.DiGraph): The SRA's memory graph.
            new_abstraction_ids (list): A list of node IDs for new abstractions to be processed.
        """
        if not new_abstraction_ids: return

        # Precompute centroids for efficient batch similarity calculation
        if self.symbols:
            centroids = np.vstack([s.centroid for s in self.symbols])
        else:
            centroids = np.empty((0, 0))

        for nid in new_abstraction_ids:
            if nid not in memory_graph.nodes: continue
            
            z = np.asarray(memory_graph.nodes[nid]["z"], dtype=np.float32)

            if not self.symbols:
                # Create the very first symbol.
                self._create_new_symbol(z, nid)
                centroids = np.vstack([s.centroid for s in self.symbols]) # Initialize centroids
                continue

            # Find the best matching existing symbol
            sims = batch_cosine_sims(centroids, z)
            best_idx = np.argmax(sims)
            best_sim = sims[best_idx]

            if best_sim >= self.cfg["assign_threshold"]:
                # Assign to existing symbol
                sym = self.symbols[best_idx]
                sym.update_with(z, nid)
                self.symbol_map[nid] = sym.id
                centroids[best_idx] = sym.centroid # Update the live centroids matrix
            else:
                # No good match found, create a new symbol
                new_sym_id = self._create_new_symbol(z, nid)
                centroids = np.vstack([centroids, self.symbols[new_sym_id].centroid])

    def _create_new_symbol(self, z: np.ndarray, node_id: str) -> int:
        """Internal helper to create and register a new symbol."""
        sym = Symbol(id=self.next_symbol_id, centroid=z.copy(), members=[node_id], count=1)
        self.symbols.append(sym)
        self.symbol_map[node_id] = sym.id
        self.next_symbol_id += 1
        return sym.id

    def build_cooccurrence_graph(self, memory_graph: nx.DiGraph, window_size: int = 10):
        """
        Analyzes the causal chain of memories to build a graph of symbol relationships.
        Symbols that appear close together in time are considered related.
        """
        if len(memory_graph) < window_size: return

        # Get the main causal chain (longest path)
        main_trail = nx.dag_longest_path(memory_graph)
        
        # Create sliding windows over the trail
        windows = [main_trail[i:i + window_size] for i in range(len(main_trail) - window_size + 1)]

        self.cooc.clear()
        for window in windows:
            # Find the unique symbols present in the current window
            window_symbols = list(set(self.symbol_map.get(nid) for nid in window if nid in self.symbol_map))
            
            # Increment co-occurrence count for every pair of symbols in the window
            for i in range(len(window_symbols)):
                for j in range(i + 1, len(window_symbols)):
                    s1, s2 = sorted((window_symbols[i], window_symbols[j]))
                    self.cooc[(s1, s2)] = self.cooc.get((s1, s2), 0) + 1
        
        # Build the graph visualization object
        self.symbol_graph = nx.Graph()
        for sym in self.symbols:
            self.symbol_graph.add_node(sym.id, count=sym.count)
        for (s1, s2), weight in self.cooc.items():
            self.symbol_graph.add_edge(s1, s2, weight=weight)

    def merge_similar_symbols(self):
        """Refines the symbolic language by merging nearly identical symbols."""
        # ... (implementation from your code, which is solid) ...
        pass # For brevity

    def prune_symbols(self):
        """Cleans up the symbolic language by removing stale or low-support symbols."""
        # ... (implementation from your code, which is solid) ...
        pass # For brevity

    def visualize_symbol_gallery(self, top_k: int = 64, cols: int = 8):
        """Visualizes the 'glyphs' for the most common symbols."""
        if self.decoder is None:
            print("Warning: No decoder provided. Cannot visualize symbol glyphs.")
            return
        if not self.symbols: return

        top_symbols = sorted(self.symbols, key=lambda s: s.count, reverse=True)[:top_k]
        rows = int(np.ceil(len(top_symbols) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.7))
        fig.suptitle("Symbol Gallery: Visual Prototypes of Core Concepts", fontsize=16)

        for i, sym in enumerate(top_symbols):
            ax = axes.flat[i] if rows > 1 else axes[i]
            glyph = self.decoder(torch.tensor(sym.centroid, dtype=torch.float32).unsqueeze(0))
            ax.imshow(glyph.detach().cpu().numpy().reshape(28, 28), cmap='gray')
            ax.set_title(f"S{sym.id} (n={sym.count})", fontsize=8)
            ax.axis('off')
        
        for i in range(len(top_symbols), len(axes.flat)):
            axes.flat[i].axis('off') # Hide unused subplots
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def visualize_symbol_graph(self, top_k_nodes: int = 50):
        """Visualizes the co-occurrence graph of symbol relationships."""
        if not self.symbol_graph:
            print("Symbol graph not built. Run \`build_cooccurrence_graph\` first.")
            return
        
        top_symbols = sorted(self.symbols, key=lambda s: s.count, reverse=True)[:top_k_nodes]
        top_symbol_ids = [s.id for s in top_symbols]
        
        subgraph = self.symbol_graph.subgraph(top_symbol_ids)
        
        node_sizes = [100 + s.count * 20 for s in top_symbols]
        edge_weights = [subgraph[u][v]['weight'] * 0.5 for u, v in subgraph.edges()]

        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(subgraph, k=0.8, iterations=50, seed=42)
        nx.draw(subgraph, pos, with_labels=True, node_color='#4da6ff', node_size=node_sizes,
                width=edge_weights, edge_color='#cccccc', font_size=9, font_weight='bold')
        plt.title("Symbol Co-occurrence Graph (Relationships Between Concepts)", fontsize=16)
        plt.show()

# -----------------------------------------------------------------------------
# Standalone Example: Simulating integration with Stage 3
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("--- SUL Manager Standalone Demonstration ---")

    # 1. Create a Fake Decoder (to visualize glyphs)
    # In a real scenario, this would be your trained autoencoder's decoder.
    decoder_model = nn.Sequential(
        nn.Linear(16, 64), nn.ReLU(),
        nn.Linear(64, 128), nn.ReLU(),
        nn.Linear(128, 28*28), nn.Sigmoid()
    )
    
    # 2. Build a Fake Memory Graph (simulating the output of Stage 3)
    memory_graph = nx.DiGraph()
    abstractions_to_process = []
    print("Simulating Stage 3: Generating a memory graph with 100 raw 'abstractions'...")
    cluster_center = np.random.randn(16) * 2
    for i in range(100):
        # Simulate some clustering in the latent space
        if i % 20 == 0:
             cluster_center = np.random.randn(16) * 2
        z = cluster_center + np.random.randn(16) * 0.5
        
        node_id = f"abs_{i}"
        memory_graph.add_node(node_id, z=z, tag='abstraction')
        if i > 0:
            memory_graph.add_edge(f"abs_{i-1}", node_id)
        abstractions_to_process.append(node_id)
    
    # 3. Initialize and Run the Symbol Manager
    print("\\nInitializing SUL...")
    sul = SymbolManager(decoder=decoder_model)

    print(f"Assigning {len(abstractions_to_process)} abstractions to symbols...")
    sul.assign_abstractions(memory_graph, abstractions_to_process)
    
    print(f"\\nDiscovered {len(sul.symbols)} unique symbols.")
    for symbol in sorted(sul.symbols, key=lambda s: s.count, reverse=True)[:5]:
        print(f"  - Symbol {symbol.id}: {symbol.count} members")

    # 4. Build and Visualize the Relationship Graph
    print("\\nBuilding co-occurrence graph from memory trail...")
    sul.build_cooccurrence_graph(memory_graph, window_size=8)
    sul.visualize_symbol_graph(top_k_nodes=20)

    # 5. Visualize the Symbol Glyphs
    print("\\nVisualizing the 'glyphs' for the most frequent symbols...")
    sul.visualize_symbol_gallery(top_k=16, cols=4)
`;

export const STAGE_5_PROTOTYPE_CODE = `
# sra_integration_demo.py
"""
End-to-End Demonstration of the complete SRA + Living Kernel Architecture.

This script integrates all four stages of our design:
1.  SRA Stage 3: Generates a rich memory graph with creative abstractions.
2.  SUL (SymbolManager): Organizes abstractions into a symbolic language.
3.  Living Kernel (Meta-Controller): Analyzes the symbolic language to
    propose a new rule for itself.
4.  Human-in-the-Loop: The user approves the self-proposed rule via the
    web dashboard, allowing the system to modify its own cognitive process.

Instructions:
1.  Make sure you have created the \`living_kernel\` package with all the
    files from the previous response.
2.  Start the web dashboard in one terminal: \`python -m living_kernel.webapp\`
3.  Run this script in another terminal: \`python sra_integration_demo.py\`
"""
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from tqdm import tqdm

# --- Import all our previously designed components ---
# Stage 3 SRA components (condensed for brevity)
from sra_stage3 import Autoencoder, RCE, SRS, TAL, MemoryGraph as SraMemoryGraph, RUS

# Stage 4 SUL
from sul_manager import SymbolManager

# Living Kernel components
from living_kernel.core import RuleStore, MetaController
from living_kernel.sandbox import SandboxRunner

# --- Configuration for the Full Demo ---
class DemoConfig:
    LATENT_DIM = 16
    SRA_EPOCHS = 2 # Run SRA for a few epochs to generate a decent memory graph
    BATCH_SIZE = 128
    # Use the same hyperparameters from sra_stage3.py
    RCE_LR = 0.1
    SRS_NOVELTY_WEIGHT = 0.3
    SRS_LOSS_WEIGHT = 0.5
    TAL_TAU = 0.80
    CONTRADICTION_THRESH = 3.5
    RUS_ITERS = 50
    RUS_LR = 0.05
    RUS_L2_REG = 1e-3

# --- Meta-Rule Generation Logic ---
def generate_meta_rule_proposal(symbol_manager: SymbolManager):
    """
    Analyzes the symbol graph to find a candidate for a new rule.
    The rule will propose unifying two frequently co-occurring symbols.
    """
    if not symbol_manager.symbol_graph or len(symbol_manager.symbol_graph.edges) == 0:
        return None, None

    # Find the edge with the highest co-occurrence weight
    try:
        s1, s2, data = sorted(symbol_manager.symbol_graph.edges(data=True),
                              key=lambda x: x[2].get('weight', 0), reverse=True)[0]
    except IndexError:
        return None, None # No edges found

    weight = data.get('weight', 0)
    print(f"\\n[Meta-Controller]: Found highly correlated symbols: S{s1} and S{s2} (co-occurrence weight: {weight})")
    print(f"[Meta-Controller]: Proposing a new rule to unify them when they appear together.")

    # --- Construct the Python code for the new rule ---
    pattern_py = f"""
def pattern(symbols, memory):
    # This rule fires if both Symbol {s1} and Symbol {s2} are present
    # in the recent context (represented here by the input \`symbols\`).
    # Note: In a real system, \`symbols\` would be a list of recent symbol objects.
    # Here, we simulate with a list of dictionaries.
    symbol_ids = {{s.get('id') for s in symbols}}
    return {s1} in symbol_ids and {s2} in symbol_ids
"""

    transform_py = f"""
def transform(symbols, memory):
    # This transform finds the vectors for Symbol {s1} and {s2}
    # and returns their average as a new, unified concept.
    vec1 = None
    vec2 = None
    for s in symbols:
        if s.get('id') == {s1}:
            vec1 = s.get('vector')
        if s.get('id') == {s2}:
            vec2 = s.get('vector')
    
    if vec1 and vec2:
        # Simple vector averaging for unification
        avg_vec = [(a + b) / 2.0 for a, b in zip(vec1, vec2)]
        return make_vector(avg_vec) # Uses the safe 'make_vector' helper
    
    return None # Return None if transform fails
"""
    return pattern_py, transform_py

# --- Main Integration Function ---
def main():
    cfg = DemoConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- SRA + SUL + Living Kernel End-to-End Demo ---")
    print(f"Using device: {device}")

    # =========================================================================
    # PART 1: Run the SRA to Generate Raw Thought (Memory Graph)
    # =========================================================================
    print("\\n--- [Part 1] Running SRA to generate a memory graph... ---")
    # Initialize SRA components
    model = Autoencoder(cfg.LATENT_DIM).to(device)
    # ... (Full SRA training loop from sra_stage3.py, condensed here)
    # NOTE: This is a simplified loop for the demo.
    sra_memory = SraMemoryGraph()
    abstractions_created = 0
    # Dummy training loop to populate memory
    for i in range(25): # Simulate a few batches
        z = np.random.randn(cfg.LATENT_DIM)
        node_id = sra_memory.add_moment(torch.tensor(z), 0, i, 0)
        if i > 0 and i % 5 == 0:
            z_star = np.random.randn(cfg.LATENT_DIM) * 2
            sra_memory.add_abstraction(torch.tensor(z_star), [sra_memory.last_node_id], 0)
            abstractions_created += 1

    print(f"SRA run complete. Generated a memory graph with {sra_memory.node_counter} nodes.")
    print(f"  - Abstractions created by RUS: {abstractions_created}")

    # =========================================================================
    # PART 2: Run the SUL to Organize Thought into a Symbolic Language
    # =========================================================================
    print("\\n--- [Part 2] Running SUL to discover symbols from the memory graph... ---")
    sul = SymbolManager(decoder=lambda z: model.decoder(z.to(device)))
    
    # Process only the abstraction nodes from the SRA memory
    abstraction_node_ids = [nid for nid, data in sra_memory.G.nodes(data=True) if data.get('tag') == 'abstraction']
    sul.assign_abstractions(sra_memory.G, abstraction_node_ids)
    sul.build_cooccurrence_graph(sra_memory.G, window_size=5)

    print(f"SUL analysis complete. Discovered {len(sul.symbols)} unique symbols.")
    sul.visualize_symbol_graph(top_k_nodes=10) # Show the relationships it found

    # =========================================================================
    # PART 3: Living Kernel Proposes a Self-Improvement Rule
    # =========================================================================
    print("\\n--- [Part 3] Living Kernel is analyzing the symbolic structure... ---")
    # Initialize Living Kernel components
    rule_store = RuleStore()
    sandbox = SandboxRunner()
    meta_controller = MetaController(rule_store, sandbox)

    # Generate a new rule proposal based on the SUL's findings
    pattern, transform = generate_meta_rule_proposal(sul)
    
    if pattern and transform:
        print("[Living Kernel]: A new rule has been proposed based on observed symbol correlations.")
        
        # The meta-controller proposes the rule, which also runs it against a test case
        test_symbols = [{'id': s.id, 'vector': s.centroid.tolist()} for s in sul.symbols[:2]]
        test_memory = {}
        
        rule, passed, logs = meta_controller.propose_from_template(
            pattern, transform, test_examples=[(test_symbols, test_memory)]
        )
        print(f"[Living Kernel]: Proposed Rule ID: {rule.id}. Sandbox test passed: {passed}")
        
        # =====================================================================
        # PART 4: Human-in-the-Loop Approval
        # =====================================================================
        print("\\n--- [Part 4] Human Approval Required ---")
        print("\\n" + "="*60)
        print("  ACTION REQUIRED:")
        print("  1. Make sure the webapp is running (\`python -m living_kernel.webapp\`).")
        print("  2. Open http://127.0.0.1:5000 in your browser.")
        print(f"  3. Find rule '{rule.id}', review its code, and click 'Approve'.")
        print("="*60 + "\\n")
        
        # Wait for the user to approve the rule
        while not rule_store.get_rule(rule.id).approved:
            try:
                input("Press Enter after you have approved the rule in the web dashboard...")
            except KeyboardInterrupt:
                print("\\nExiting demo.")
                return
        
        print("\\n[Living Kernel]: Rule approved by human operator! The new cognitive rule is now active.")

        # =====================================================================
        # PART 5: Applying the New, Self-Generated Rule
        # =====================================================================
        print("\\n--- [Part 5] Running a new cognitive cycle with the approved rule... ---")
        # Simulate a new context where the two correlated symbols appear
        s1_id = int(rule.pattern_py.split("in symbol_ids and ")[0].split(" ")[-1])
        s2_id = int(rule.pattern_py.split("in symbol_ids and ")[1].split(" ")[0])
        
        s1_vec = next(s.centroid.tolist() for s in sul.symbols if s.id == s1_id)
        s2_vec = next(s.centroid.tolist() for s in sul.symbols if s.id == s2_id)
        
        new_context_symbols = [{'id': s1_id, 'vector': s1_vec}, {'id': s2_id, 'vector': s2_vec}]
        
        approved_rule = rule_store.get_rule(rule.id)
        ok, out = sandbox.run_rule_once(approved_rule, new_context_symbols, {})
        
        print(f"\\nApplying approved rule '{approved_rule.id}' to a new context...")
        print(f"  - Sandbox execution OK: {ok}")
        print(f"  - Pattern matched: {out.get('pattern')}")
        print(f"  - Transform output (new unified vector): {out.get('transform')}")
        print("\\n--- Demo Complete ---")
        print("The system successfully identified a pattern in its own thoughts, proposed a rule to act on it, and applied that rule after human approval.")

    else:
        print("[Living Kernel]: No strong symbol correlations found. No new rule proposed.")

if __name__ == "__main__":
    main()
`;
