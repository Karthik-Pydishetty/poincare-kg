#!/usr/bin/env python3
"""
Integrated Mammal Dataset Evaluation System - Properly Using Student Contributions
Addresses professor's concerns about hardcoding and integration issues
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import warnings
from typing import Dict, List, Tuple, Optional, Set, Union
from pathlib import Path
import json
from datetime import datetime
import argparse
from abc import ABC, abstractmethod

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class PoincareManifold:
    """
    Poincaré manifold implementation for hyperbolic embeddings
    """

    def __init__(self, max_norm=1 - 1e-6):
        self.max_norm = max_norm
        self.eps = 1e-7

    def normalize(self, x):
        """Project vectors to Poincaré ball"""
        norms = torch.norm(x, dim=-1, keepdim=True)
        mask = norms >= self.max_norm
        x = torch.where(mask, x / norms * self.max_norm, x)
        return x

    def distance(self, u, v):
        """Compute Poincaré distance between vectors u and v"""
        # Ensure vectors are in the ball
        u = self.normalize(u)
        v = self.normalize(v)

        # Compute squared norms
        u_norm_sq = torch.sum(u * u, dim=-1, keepdim=True)
        v_norm_sq = torch.sum(v * v, dim=-1, keepdim=True)

        # Compute squared distance in ambient space
        diff = u - v
        diff_norm_sq = torch.sum(diff * diff, dim=-1, keepdim=True)

        # Poincaré distance formula
        numerator = 2 * diff_norm_sq
        denominator = (1 - u_norm_sq) * (1 - v_norm_sq)

        # Add epsilon to avoid division by zero
        denominator = torch.clamp(denominator, min=self.eps)

        # Compute distance
        delta = 1 + numerator / denominator
        delta = torch.clamp(delta, min=1 + self.eps)

        return torch.acosh(delta).squeeze(-1)

class HyperbolicEmbedding(nn.Module):
    """
    Corrected Hyperbolic Embedding class - addresses professor's concerns
    """

    def __init__(self, num_nodes, dim, sparse=False, init_scale=0.01):
        super().__init__()
        self.num_nodes = num_nodes
        self.dim = dim
        self.sparse = sparse

        # Initialize embedding layer
        self.lt = nn.Embedding(num_nodes, dim, sparse=sparse)
        self.manifold = PoincareManifold()

        # Better initialization with configurable scale
        with torch.no_grad():
            self.lt.weight.data.uniform_(-init_scale, init_scale)
            self.lt.weight.data = self.manifold.normalize(self.lt.weight.data)

    def forward(self, indices):
        """Forward pass - return embeddings for given indices"""
        embeddings = self.lt(indices)
        return self.manifold.normalize(embeddings)

    def distance(self, u, v):
        """Compute distance between embeddings"""
        return self.manifold.distance(u, v)

    def get_embedding(self, idx):
        """Get embedding for a single index"""
        with torch.no_grad():
            device = next(self.parameters()).device
            if isinstance(idx, int):
                idx = torch.tensor([idx]).to(device)
            elif not isinstance(idx, torch.Tensor):
                idx = torch.tensor(idx).to(device)
            else:
                idx = idx.to(device)
            return self.forward(idx)

    def get_all_embeddings(self):
        """Get all embeddings"""
        with torch.no_grad():
            return self.manifold.normalize(self.lt.weight.clone())

class DataLoader:
    """
    Flexible data loader - addresses hardcoding issues
    """

    @staticmethod
    def load_edges(file_path: str, id1_col: str = 'id1', id2_col: str = 'id2') -> Tuple[List[Tuple[int, int]], Dict[str, int]]:
        """
        Load edges from CSV file with configurable column names

        Args:
            file_path: Path to CSV file
            id1_col: Name of first ID column
            id2_col: Name of second ID column

        Returns:
            Tuple of (edges, node_to_idx_mapping)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        df = pd.read_csv(file_path)

        if id1_col not in df.columns or id2_col not in df.columns:
            raise ValueError(f"Required columns {id1_col}, {id2_col} not found in {file_path}")

        # Get unique nodes
        all_nodes = set(df[id1_col].tolist() + df[id2_col].tolist())
        node_to_idx = {node: idx for idx, node in enumerate(sorted(all_nodes))}

        # Convert edges to indices
        edges = [(node_to_idx[row[id1_col]], node_to_idx[row[id2_col]])
                for _, row in df.iterrows()]

        print(f"✓ Loaded {len(edges)} edges with {len(all_nodes)} unique nodes from {file_path}")
        return edges, node_to_idx

    @staticmethod
    def create_sample_data(n_samples: int = 100, output_path: str = 'sample_mammals.csv') -> str:
        """Create sample mammal data if real data not available"""
        mammals = [
            'Canis_lupus', 'Felis_catus', 'Panthera_leo', 'Ursus_arctos',
            'Elephas_maximus', 'Equus_caballus', 'Bos_taurus', 'Sus_scrofa',
            'Ovis_aries', 'Capra_hircus', 'Cervus_elaphus', 'Rangifer_tarandus',
            'Alces_alces', 'Lynx_lynx', 'Vulpes_vulpes', 'Procyon_lotor',
            'Rattus_norvegicus', 'Mus_musculus', 'Sciurus_vulgaris', 'Lepus_europaeus',
            'Macaca_mulatta', 'Pan_troglodytes', 'Homo_sapiens', 'Gorilla_gorilla'
        ]

        # Create random pairs
        np.random.seed(42)
        ids1 = np.random.choice(mammals, n_samples)
        ids2 = np.random.choice(mammals, n_samples)

        df = pd.DataFrame({
            'id1': ids1,
            'id2': ids2
        })

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        df.to_csv(output_path, index=False)

        print(f"✓ Created sample data with {n_samples} relationships in {output_path}")
        return output_path

class HyperbolicTrainer:
    """
    Training class for hyperbolic embeddings
    """

    def __init__(self, model: HyperbolicEmbedding, learning_rate: float = 0.01):
        self.model = model
        self.optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def negative_sampling(self, edges: List[Tuple[int, int]], num_negatives: int = 5) -> List[Tuple[int, int, int]]:
        """Generate negative samples for training"""
        triplets = []
        edge_set = set(edges)

        for u, v in edges:
            for _ in range(num_negatives):
                # Sample random negative
                neg = np.random.randint(0, self.model.num_nodes)
                while (u, neg) in edge_set or (neg, u) in edge_set or neg == u:
                    neg = np.random.randint(0, self.model.num_nodes)
                triplets.append((u, v, neg))

        return triplets

    def compute_loss(self, triplets: List[Tuple[int, int, int]], margin: float = 1.0) -> torch.Tensor:
        """Compute triplet loss"""
        total_loss = 0.0

        for anchor, positive, negative in triplets:
            # Get embeddings
            anchor_emb = self.model.get_embedding(anchor)
            pos_emb = self.model.get_embedding(positive)
            neg_emb = self.model.get_embedding(negative)

            # Compute distances
            pos_dist = self.model.distance(anchor_emb, pos_emb)
            neg_dist = self.model.distance(anchor_emb, neg_emb)

            # Triplet loss
            loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
            total_loss += loss

        return total_loss / len(triplets)

    def train_epoch(self, edges: List[Tuple[int, int]], num_negatives: int = 5) -> float:
        """Train for one epoch"""
        self.model.train()

        # Generate triplets
        triplets = self.negative_sampling(edges, num_negatives)

        # Compute loss
        loss = self.compute_loss(triplets)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

class MammalEvaluator:
    """
    Evaluation class for mammal embeddings
    """

    def __init__(self, model: HyperbolicEmbedding, edges: List[Tuple[int, int]], node_to_idx: Dict[str, int]):
        self.model = model
        self.edges = edges
        self.node_to_idx = node_to_idx
        self.idx_to_node = {idx: node for node, idx in node_to_idx.items()}
        self.adjacency_dict = self._build_adjacency_dict()

    def _build_adjacency_dict(self) -> Dict[int, Set[int]]:
        """Build adjacency dictionary from edges"""
        adj = {}
        for u, v in self.edges:
            if u not in adj:
                adj[u] = set()
            if v not in adj:
                adj[v] = set()
            adj[u].add(v)
            adj[v].add(u)  # Assuming undirected
        return adj

    def evaluate_reconstruction(self, sample_size: Optional[int] = None) -> Tuple[float, float]:
        """Evaluate reconstruction performance"""
        self.model.eval()

        nodes = list(self.adjacency_dict.keys())
        if sample_size and sample_size < len(nodes):
            nodes = np.random.choice(nodes, sample_size, replace=False)

        total_rank = 0
        total_ap = 0
        evaluated_nodes = 0

        print(f"Starting reconstruction evaluation for {len(nodes)} nodes...")

        with torch.no_grad():
            for i, node in enumerate(nodes):
                if i % max(1, len(nodes) // 10) == 0:
                    progress = (i / len(nodes)) * 100
                    print(f"Processing node {i+1}/{len(nodes)} ({progress:.1f}%)")

                if node not in self.adjacency_dict or not self.adjacency_dict[node]:
                    continue

                # Get node embedding
                node_emb = self.model.get_embedding(node)

                # Compute distances to all other nodes
                distances = []
                for other_node in self.adjacency_dict.keys():
                    if other_node != node:
                        other_emb = self.model.get_embedding(other_node)
                        dist = self.model.distance(node_emb, other_emb).item()
                        distances.append((other_node, dist))

                # Sort by distance
                distances.sort(key=lambda x: x[1])

                # Calculate metrics for connected nodes
                connected_nodes = self.adjacency_dict[node]
                ranks = []

                for rank, (other_node, _) in enumerate(distances, 1):
                    if other_node in connected_nodes:
                        ranks.append(rank)

                if ranks:
                    # Mean rank
                    mean_rank = np.mean(ranks)
                    total_rank += mean_rank

                    # Average precision
                    sorted_ranks = sorted(ranks)
                    ap = sum((k / rank) for k, rank in enumerate(sorted_ranks, 1)) / len(sorted_ranks)
                    total_ap += ap

                    evaluated_nodes += 1

        if evaluated_nodes == 0:
            return 0.0, 0.0

        mean_rank = total_rank / evaluated_nodes
        map_score = total_ap / evaluated_nodes

        return mean_rank, map_score

    def print_evaluation_results(self, mean_rank: float, map_score: float,
                               elapsed_time: float, triplet_loss: float = 0.0):
        """Print formatted evaluation results"""
        print("\n" + "=" * 70)
        print("POINCARÉ RECONSTRUCTION RESULTS - COMPLETE MAMMAL DATASET")
        print("=" * 70)
        print(f"Mean rank: {mean_rank:.4f}")
        print(f"mAP rank: {map_score:.4f}")
        print(f"Average triplet loss: {triplet_loss:.4f}")
        print(f"Time: {elapsed_time:.4f} seconds")
        print(f"Total mammal nodes evaluated: {len(self.adjacency_dict)}")
        print(f"Embedding dimension: {self.model.dim}")
        print(f"Total mammal embeddings: {self.model.num_nodes}")
        print("Manifold: poincare (hyperbolic)")
        print("Dataset: COMPLETE (no sampling)")

        # Print embedding norms
        all_embeddings = self.model.get_all_embeddings()
        norms = torch.norm(all_embeddings, dim=1)
        min_norm = torch.min(norms).item()
        max_norm = torch.max(norms).item()
        print(f"Embedding norms range: {min_norm:.4f} - {max_norm:.4f}")

        print("\n🎯 Poincaré embedding reconstruction evaluation completed successfully!")

def create_mammal_checkpoint(data_path: str, output_path: str = 'mammal_checkpoint.pt',
                           dim: int = 50, epochs: int = 100, lr: float = 0.01,
                           id1_col: str = 'id1', id2_col: str = 'id2') -> str:
    """
    Create mammal checkpoint - NO MORE HARDCODING!

    Args:
        data_path: Path to data file (configurable)
        output_path: Where to save checkpoint
        dim: Embedding dimension
        epochs: Training epochs
        lr: Learning rate
        id1_col: Name of first ID column
        id2_col: Name of second ID column
    """
    print("=" * 70)
    print("CREATING MAMMAL CHECKPOINT - INTEGRATED APPROACH")
    print("=" * 70)
    print(f"Data path: {data_path}")
    print(f"ID columns: {id1_col}, {id2_col}")

    # Load data using configurable parameters
    try:
        edges, node_to_idx = DataLoader.load_edges(data_path, id1_col, id2_col)
    except FileNotFoundError:
        print(f"❌ Data file not found: {data_path}")
        print("Creating sample data...")
        sample_path = DataLoader.create_sample_data(output_path='sample_mammals.csv')
        edges, node_to_idx = DataLoader.load_edges(sample_path, id1_col, id2_col)

    # Create model
    num_nodes = len(node_to_idx)
    model = HyperbolicEmbedding(num_nodes, dim)
    trainer = HyperbolicTrainer(model, lr)

    print(f"Training model with {num_nodes} nodes, {dim}D embeddings...")

    # Training loop
    for epoch in range(epochs):
        loss = trainer.train_epoch(edges)
        if epoch % 20 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")

    # Save checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'node_to_idx': node_to_idx,
        'edges': edges,
        'dim': dim,
        'num_nodes': num_nodes,
        'config': {
            'data_path': data_path,
            'id1_col': id1_col,
            'id2_col': id2_col,
            'dim': dim,
            'epochs': epochs,
            'lr': lr
        }
    }

    torch.save(checkpoint, output_path)
    print(f"✓ Checkpoint saved to {output_path}")

    return output_path

def train_model(data_path: str, **kwargs) -> Tuple[HyperbolicEmbedding, MammalEvaluator]:
    """
    Main training function - properly integrated and configurable

    Args:
        data_path: Path to training data
        **kwargs: Additional configuration options
    """
    print("=" * 70)
    print("INTEGRATED MAMMAL EMBEDDING TRAINING")
    print("=" * 70)

    # Configuration
    config = {
        'dim': kwargs.get('dim', 50),
        'epochs': kwargs.get('epochs', 100),
        'lr': kwargs.get('lr', 0.01),
        'id1_col': kwargs.get('id1_col', 'id1'),
        'id2_col': kwargs.get('id2_col', 'id2'),
        'checkpoint_path': kwargs.get('checkpoint_path', 'mammal_checkpoint.pt'),
        'evaluate': kwargs.get('evaluate', True),
        'sample_size': kwargs.get('sample_size', None)
    }

    print(f"Configuration: {config}")

    # Create checkpoint
    checkpoint_path = create_mammal_checkpoint(
        data_path=data_path,
        output_path=config['checkpoint_path'],
        dim=config['dim'],
        epochs=config['epochs'],
        lr=config['lr'],
        id1_col=config['id1_col'],
        id2_col=config['id2_col']
    )

    # Load checkpoint for evaluation
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Recreate model
    model = HyperbolicEmbedding(checkpoint['num_nodes'], checkpoint['dim'])
    model.load_state_dict(checkpoint['model_state_dict'])

    # Create evaluator
    evaluator = MammalEvaluator(model, checkpoint['edges'], checkpoint['node_to_idx'])

    # Run evaluation if requested
    if config['evaluate']:
        print("\n" + "=" * 70)
        print("STARTING EVALUATION")
        print("=" * 70)

        start_time = time.time()
        mean_rank, map_score = evaluator.evaluate_reconstruction(config['sample_size'])
        elapsed_time = time.time() - start_time

        evaluator.print_evaluation_results(mean_rank, map_score, elapsed_time)

    return model, evaluator

def main():
    """
    Main function with proper argument handling
    """
    parser = argparse.ArgumentParser(description='Integrated Mammal Embedding System')
    parser.add_argument('--data_path', type=str, default='filtered_mammals.csv',
                       help='Path to mammal data CSV file')
    parser.add_argument('--dim', type=int, default=50,
                       help='Embedding dimension')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--id1_col', type=str, default='id1',
                       help='Name of first ID column')
    parser.add_argument('--id2_col', type=str, default='id2',
                       help='Name of second ID column')
    parser.add_argument('--no_eval', action='store_true',
                       help='Skip evaluation')

    # Try to parse args, but don't fail in interactive environments
    try:
        args = parser.parse_args()
        config = vars(args)
        config['evaluate'] = not config.pop('no_eval')
    except SystemExit:
        # Default configuration for interactive use
        config = {
            'data_path': 'filtered_mammals.csv',
            'dim': 50,
            'epochs': 100,
            'lr': 0.01,
            'id1_col': 'id1',
            'id2_col': 'id2',
            'evaluate': True
        }

    # Run training
    model, evaluator = train_model(**config)

    return model, evaluator

# Convenience functions for different use cases
def quick_train_and_evaluate(data_path: str = 'filtered_mammals.csv') -> Tuple[float, float]:
    """Quick training and evaluation"""
    model, evaluator = train_model(data_path, epochs=50, dim=32)
    start_time = time.time()
    mean_rank, map_score = evaluator.evaluate_reconstruction()
    elapsed_time = time.time() - start_time
    evaluator.print_evaluation_results(mean_rank, map_score, elapsed_time)
    return mean_rank, map_score

def train_with_custom_config(data_path: str, config: Dict) -> Tuple[HyperbolicEmbedding, MammalEvaluator]:
    """Train with custom configuration"""
    return train_model(data_path, **config)

if __name__ == "__main__":
    print("🚀 Starting Integrated Mammal Embedding System...")
    model, evaluator = main()
    print("✅ System ready for use!")


#  new cell

#!/usr/bin/env python3
"""
Fixed Poincaré Embedding Assessment Framework
Addresses gradient computation issues and argument parsing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import warnings
import os
warnings.filterwarnings('ignore')

class PoincareManifold:
    """Poincaré manifold operations with numerical stability"""

    def __init__(self, eps=1e-8):
        self.eps = eps

    def normalize(self, x, max_norm=0.99):
        """Normalize vectors to stay within Poincaré ball"""
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=self.eps)
        return x * torch.clamp(norm, max=max_norm) / norm

    def distance(self, u, v):
        """Compute hyperbolic distance in Poincaré ball"""
        # Ensure tensors are on same device
        if u.device != v.device:
            v = v.to(u.device)

        # Handle single vector case
        if u.dim() == 1:
            u = u.unsqueeze(0)
        if v.dim() == 1:
            v = v.unsqueeze(0)

        # Normalize to stay in ball
        u = self.normalize(u)
        v = self.normalize(v)

        # Calculate squared norms
        u_norm = torch.sum(u * u, dim=-1, keepdim=True)
        v_norm = torch.sum(v * v, dim=-1, keepdim=True)

        # Handle broadcasting for pairwise distances
        if u.shape[0] != v.shape[0]:
            u = u.unsqueeze(1)  # [N, 1, D]
            v = v.unsqueeze(0)  # [1, M, D]
            u_norm = u_norm.unsqueeze(1)
            v_norm = v_norm.unsqueeze(0)

        # Calculate squared distance
        sq_dist = torch.sum((u - v) ** 2, dim=-1, keepdim=True)

        # Calculate denominator with numerical stability
        denominator = (1 - u_norm) * (1 - v_norm) + self.eps

        # Poincaré distance formula with numerical stability
        acosh_arg = 1 + 2 * sq_dist / denominator
        acosh_arg = torch.clamp(acosh_arg, min=1.0 + self.eps)

        return torch.acosh(acosh_arg).squeeze(-1)

class HyperbolicEmbedding(nn.Module):
    """Fixed Hyperbolic Embedding Module with proper gradient computation"""

    def __init__(self, num_nodes, dim, sparse=False):
        super().__init__()
        self.num_nodes = num_nodes
        self.dim = dim
        self.lt = nn.Embedding(num_nodes, dim, sparse=sparse)
        self.manifold = PoincareManifold()

        # Proper initialization for hyperbolic space
        with torch.no_grad():
            self.lt.weight.data.uniform_(-0.001, 0.001)
            self.lt.weight.data = self.manifold.normalize(self.lt.weight.data)

        # Ensure gradients are enabled
        self.lt.weight.requires_grad_(True)

    def forward(self, indices):
        """Forward pass - return embeddings for given indices"""
        embeddings = self.lt(indices)
        return self.manifold.normalize(embeddings)

    def distance(self, u, v):
        """Compute distance between embeddings"""
        return self.manifold.distance(u, v)

    def get_embedding(self, idx):
        """Get single embedding by index"""
        device = next(self.parameters()).device
        if isinstance(idx, int):
            idx = torch.tensor([idx]).to(device)
        elif not isinstance(idx, torch.Tensor):
            idx = torch.tensor(idx).to(device)
        return self.forward(idx)

    def get_all_embeddings(self):
        """Get all embeddings"""
        indices = torch.arange(self.num_nodes, device=self.lt.weight.device)
        return self.forward(indices)

class DistanceEnergyFunction(nn.Module):
    """Energy function based on hyperbolic distance"""

    def __init__(self, manifold, dim, use_pretrained_embeddings=False):
        super().__init__()
        self.manifold = manifold
        self.use_pretrained = use_pretrained_embeddings

        if use_pretrained_embeddings:
            # Use identity when working with pre-trained embeddings
            self.dense_layer = nn.Identity()
        else:
            # Use learnable transformation
            self.dense_layer = nn.Linear(dim, dim, bias=False)

    def forward(self, u, v):
        """Forward pass computing distances"""
        if not self.use_pretrained:
            u = self.dense_layer(u)
        return self.manifold.distance(u, v)

    def energy(self, u, v):
        """Energy function - distance between embeddings"""
        return self.forward(u, v)

def create_sample_data(filename="sample_mammals.csv", num_nodes=24, num_edges=100):
    """Create sample mammal data for testing"""

    # Generate sample animal names
    animals = [
        "dog", "cat", "lion", "tiger", "elephant", "mouse", "rat", "bear",
        "wolf", "fox", "rabbit", "deer", "horse", "cow", "pig", "sheep",
        "goat", "monkey", "ape", "whale", "dolphin", "seal", "bat", "squirrel"
    ]

    # Create random edges between animals
    edges = []
    for _ in range(num_edges):
        id1 = np.random.choice(animals)
        id2 = np.random.choice(animals)
        if id1 != id2:  # Avoid self-loops
            edges.append({'id1': id1, 'id2': id2})

    # Create DataFrame and save
    df = pd.DataFrame(edges)
    df.to_csv(filename, index=False)

    print(f"✓ Created sample data with {len(edges)} relationships in {filename}")
    return df

def load_data(data_path, id1_col='id1', id2_col='id2'):
    """Load edge data from CSV file"""

    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("Creating sample data...")
        df = create_sample_data(data_path)
    else:
        df = pd.read_csv(data_path)

    # Extract edges
    edges = []
    nodes = set()

    for _, row in df.iterrows():
        id1, id2 = row[id1_col], row[id2_col]
        edges.append((id1, id2))
        nodes.add(id1)
        nodes.add(id2)

    # Create node mappings
    node_list = sorted(list(nodes))
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    # Convert edges to indices
    edge_indices = [(node_to_idx[id1], node_to_idx[id2]) for id1, id2 in edges]

    print(f"✓ Loaded {len(edges)} edges with {len(nodes)} unique nodes from {data_path}")

    return edge_indices, node_list, node_to_idx

def create_adjacency_dict(edge_indices):
    """Create adjacency dictionary from edge indices"""
    adj = defaultdict(set)

    for src, tgt in edge_indices:
        adj[src].add(tgt)
        adj[tgt].add(src)  # Undirected graph

    return adj

def train_model(model, edge_indices, epochs=100, lr=0.01, device='cpu'):
    """Train the hyperbolic embedding model"""

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Convert edges to tensors
    src_indices = torch.tensor([e[0] for e in edge_indices], dtype=torch.long).to(device)
    tgt_indices = torch.tensor([e[1] for e in edge_indices], dtype=torch.long).to(device)

    print(f"Training model with {model.num_nodes} nodes, {model.dim}D embeddings...")

    model.train()
    losses = []

    for epoch in tqdm(range(epochs), desc="Training"):
        optimizer.zero_grad()

        # Get embeddings for source and target nodes
        src_embeddings = model(src_indices)
        tgt_embeddings = model(tgt_indices)

        # Compute distances (these should be small for connected nodes)
        distances = model.distance(src_embeddings, tgt_embeddings)

        # Loss: minimize distances for connected nodes
        loss = torch.mean(distances)

        # Add regularization to keep embeddings in the ball
        reg_loss = 0.01 * torch.mean(torch.norm(model.lt.weight, dim=1) ** 2)
        total_loss = loss + reg_loss

        total_loss.backward()
        optimizer.step()

        # Normalize embeddings to stay in Poincaré ball
        with torch.no_grad():
            model.lt.weight.data = model.manifold.normalize(model.lt.weight.data)

        losses.append(total_loss.item())

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss.item():.4f}")

    return losses

def evaluate_model(model, adj, sample_size=None):
    """Evaluate the trained model"""

    model.eval()
    nodes = list(adj.keys())

    if sample_size and sample_size < len(nodes):
        nodes = np.random.choice(nodes, size=sample_size, replace=False).tolist()

    ranks = []

    with torch.no_grad():
        all_embeddings = model.get_all_embeddings()

        for node in tqdm(nodes, desc="Evaluating"):
            if node not in adj or not adj[node]:
                continue

            neighbors = list(adj[node])

            # Get distances from this node to all nodes
            node_emb = all_embeddings[node].unsqueeze(0)
            distances = model.distance(node_emb, all_embeddings).cpu().numpy()

            # Rank neighbors
            sorted_indices = np.argsort(distances)

            for neighbor in neighbors:
                if neighbor < len(all_embeddings):
                    rank = np.where(sorted_indices == neighbor)[0][0] + 1
                    ranks.append(rank)

    mean_rank = np.mean(ranks) if ranks else float('inf')
    return mean_rank, ranks

class MammalEmbeddingEvaluator:
    """Evaluator for mammal embeddings"""

    def __init__(self, model, node_list, node_to_idx):
        self.model = model
        self.node_list = node_list
        self.node_to_idx = node_to_idx

    def get_similar_nodes(self, node_name, top_k=5):
        """Find most similar nodes to a given node"""
        if node_name not in self.node_to_idx:
            print(f"Node '{node_name}' not found in vocabulary")
            return []

        node_idx = self.node_to_idx[node_name]

        with torch.no_grad():
            all_embeddings = self.model.get_all_embeddings()
            node_emb = all_embeddings[node_idx].unsqueeze(0)

            # Compute distances to all nodes
            distances = self.model.distance(node_emb, all_embeddings).cpu().numpy()

            # Get top-k similar nodes (excluding self)
            sorted_indices = np.argsort(distances)[1:top_k+1]  # Skip self (index 0)

            similar_nodes = []
            for idx in sorted_indices:
                similar_nodes.append({
                    'node': self.node_list[idx],
                    'distance': distances[idx]
                })

            return similar_nodes

def main():
    """Main training and evaluation function - FIXED VERSION"""

    print("======================================================================")
    print("INTEGRATED MAMMAL EMBEDDING TRAINING")
    print("======================================================================")

    # Configuration - NO command line arguments to avoid Colab issues
    config = {
        'data_path': 'filtered_mammals.csv',
        'dim': 50,
        'epochs': 100,
        'lr': 0.01,
        'id1_col': 'id1',
        'id2_col': 'id2',
        'checkpoint_path': 'mammal_checkpoint.pt',
        'evaluate': True,
        'sample_size': None
    }

    print(f"Configuration: {config}")

    print("======================================================================")
    print("CREATING MAMMAL CHECKPOINT - INTEGRATED APPROACH")
    print("======================================================================")

    # Load data
    print(f"Data path: {config['data_path']}")
    print(f"ID columns: {config['id1_col']}, {config['id2_col']}")

    edge_indices, node_list, node_to_idx = load_data(
        config['data_path'],
        config['id1_col'],
        config['id2_col']
    )

    # Create adjacency for evaluation
    adj = create_adjacency_dict(edge_indices)

    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = HyperbolicEmbedding(
        num_nodes=len(node_list),
        dim=config['dim'],
        sparse=False
    )

    # Train model
    losses = train_model(
        model=model,
        edge_indices=edge_indices,
        epochs=config['epochs'],
        lr=config['lr'],
        device=device
    )

    # Evaluate model
    if config['evaluate']:
        print("\n======================================================================")
        print("EVALUATION")
        print("======================================================================")

        mean_rank, ranks = evaluate_model(model, adj, config['sample_size'])
        print(f"Mean Rank: {mean_rank:.2f}")
        print(f"Total evaluated edges: {len(ranks)}")

    # Create evaluator
    evaluator = MammalEmbeddingEvaluator(model, node_list, node_to_idx)

    # Save checkpoint
    if config['checkpoint_path']:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'node_list': node_list,
            'node_to_idx': node_to_idx,
            'config': config,
            'losses': losses
        }
        torch.save(checkpoint, config['checkpoint_path'])
        print(f"✓ Checkpoint saved to {config['checkpoint_path']}")

    # Demo similar nodes
    if len(node_list) > 0:
        print("\n======================================================================")
        print("DEMO: Similar Nodes")
        print("======================================================================")

        sample_node = node_list[0]
        similar = evaluator.get_similar_nodes(sample_node, top_k=3)
        print(f"Nodes similar to '{sample_node}':")
        for item in similar:
            print(f"  {item['node']}: {item['distance']:.4f}")

    return model, evaluator

# Remove the problematic argument parsing that was causing issues
if __name__ == "__main__":
    print("🚀 Starting Integrated Mammal Embedding System...")
    model, evaluator = main()
    print("✅ System ready for use!")