# Install torch if needed
# !pip install torch --quiet

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ========== Step 1: Create Sample Mammal Dataset ==========
def create_mammal_checkpoint(filepath='filtered_mammals.csv'):
    sample_mammals = {
        'id1': ['mammal.n.01', 'carnivore.n.01', 'cat.n.01', 'dog.n.01', 'primate.n.02'],
        'id2': ['carnivore.n.01', 'cat.n.01', 'tiger.n.02', 'wolf.n.01', 'human.n.01']
    }
    pd.DataFrame(sample_mammals).to_csv(filepath, index=False, header=False)
    print(f"✓ Dataset saved to {filepath}")

# ========== Step 2: Poincare Ball Manifold ==========
class PoincareManifold:
    def __init__(self, eps=1e-5):  # Increased from 1e-12 to prevent numerical issues
        self.eps = eps

    def normalize(self, u):
        norm = torch.norm(u, dim=-1, keepdim=True)
        max_norm = 1 - self.eps
        scale = torch.where(norm > max_norm, max_norm / (norm + self.eps), torch.ones_like(norm))
        return u * scale

    def distance(self, u, v):
        # More numerically stable distance computation
        delta = u - v
        delta_norm_sq = torch.sum(delta**2, dim=-1)
        u_norm_sq = torch.sum(u**2, dim=-1).clamp(max=1 - self.eps)
        v_norm_sq = torch.sum(v**2, dim=-1).clamp(max=1 - self.eps)

        # Prevent division by zero and numerical instability
        denominator = (1 - u_norm_sq) * (1 - v_norm_sq)
        denominator = torch.clamp(denominator, min=self.eps)

        ratio = 2 * delta_norm_sq / denominator
        # Clamp to prevent numerical issues in acosh
        arg = torch.clamp(1 + ratio, min=1 + self.eps, max=1e6)
        return torch.acosh(arg)

class RiemannianSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.data.add_(p.grad.data, alpha=-group['lr'])

class HyperbolicEmbedding(nn.Module):
    def __init__(self, num_nodes, dim, sparse=False):
        super().__init__()
        self.lt = nn.Embedding(num_nodes, dim, sparse=sparse)
        self.manifold = PoincareManifold()
        # Better initialization - slightly larger initial values
        self.lt.weight.data.uniform_(-0.001, 0.001)
        self.lt.weight.data = self.manifold.normalize(self.lt.weight.data)

    def forward(self, edges):
        return self.lt(edges)

    def distance(self, u, v):
        return self.manifold.distance(u, v)

    def get_embedding(self, idx):
        with torch.no_grad():
            device = next(self.parameters()).device
            return self.manifold.normalize(self.lt(torch.tensor([idx]).to(device)))

    def get_all_embeddings(self):
        with torch.no_grad():
            return self.manifold.normalize(self.lt.weight.clone())

class GraphDataset:
    def __init__(self, edges, num_nodes, batch_size=10, num_negs=50):
        self.edges = edges
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.num_negs = num_negs
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.edges):
            self.index = 0
            raise StopIteration

        end = min(self.index + self.batch_size, len(self.edges))
        pos = self.edges[self.index:end]
        neg = [[np.random.randint(self.num_nodes), np.random.randint(self.num_nodes)]
               for _ in range(len(pos) * self.num_negs)]
        self.index = end
        return torch.tensor(pos), torch.tensor(neg)

def load_edges(path):
    edges, nodes = [], set()
    with open(path, 'r') as f:
        for line in f:
            u, v = line.strip().split(',')
            edges.append((u, v))
            edges.append((v, u))  # symmetrize
            nodes.update([u, v])

    node_to_idx = {node: i for i, node in enumerate(sorted(nodes))}
    idx_edges = [[node_to_idx[u], node_to_idx[v]] for u, v in edges]
    return idx_edges, node_to_idx

def hinge_loss(pos_dist, neg_dist, margin=0.1):
    return F.relu(margin + pos_dist - neg_dist).mean()

# ========== NEW: Evaluation Metrics Class ==========
class EmbeddingEvaluator:
    def __init__(self, node_to_idx):
        self.node_to_idx = node_to_idx
        self.idx_to_node = {v: k for k, v in node_to_idx.items()}
        self.num_nodes = len(node_to_idx)
        self.create_ground_truth()

    def create_ground_truth(self):
        """Create hierarchical relationships for evaluation"""
        # Define the mammal taxonomy hierarchy
        hierarchy = {
            'mammal.n.01': ['carnivore.n.01', 'primate.n.02'],
            'carnivore.n.01': ['cat.n.01', 'dog.n.01'],
            'cat.n.01': ['tiger.n.02'],
            'primate.n.02': ['human.n.01'],
            'dog.n.01': ['wolf.n.01']
        }

        # Create positive pairs (parent-child and sibling relationships)
        self.positive_pairs = set()

        # Add parent-child relationships
        for parent, children in hierarchy.items():
            if parent in self.node_to_idx:
                parent_idx = self.node_to_idx[parent]
                for child in children:
                    if child in self.node_to_idx:
                        child_idx = self.node_to_idx[child]
                        self.positive_pairs.add((parent_idx, child_idx))
                        self.positive_pairs.add((child_idx, parent_idx))  # bidirectional

        # Add sibling relationships
        for parent, children in hierarchy.items():
            valid_children = [c for c in children if c in self.node_to_idx]
            for i, child1 in enumerate(valid_children):
                for child2 in valid_children[i+1:]:
                    idx1, idx2 = self.node_to_idx[child1], self.node_to_idx[child2]
                    self.positive_pairs.add((idx1, idx2))
                    self.positive_pairs.add((idx2, idx1))

        # Create negative pairs
        all_pairs = set()
        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):
                all_pairs.add((i, j))
                all_pairs.add((j, i))

        self.negative_pairs = all_pairs - self.positive_pairs

    def compute_distance_matrix(self, embeddings):
        """Compute pairwise distance matrix using Poincaré distance"""
        n = embeddings.shape[0]
        distance_matrix = torch.zeros(n, n)
        manifold = PoincareManifold()

        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = manifold.distance(
                        embeddings[i:i+1],
                        embeddings[j:j+1]
                    )
                    distance_matrix[i, j] = dist.item()

        return distance_matrix

    def evaluate_map(self, embeddings, k_values=[1, 3, 5]):
        """Compute mean Average Precision (mAP) at different k values"""
        distance_matrix = self.compute_distance_matrix(embeddings)

        # Convert distances to similarities (smaller distance = higher similarity)
        max_dist = distance_matrix.max()
        similarity_matrix = max_dist - distance_matrix

        map_scores = {}

        for k in k_values:
            average_precisions = []

            for query_idx in range(self.num_nodes):
                # Get similarities for this query
                similarities = similarity_matrix[query_idx].clone()

                # Create ground truth labels
                labels = torch.zeros(self.num_nodes)
                for target_idx in range(self.num_nodes):
                    if (query_idx, target_idx) in self.positive_pairs:
                        labels[target_idx] = 1

                # Remove self-similarity
                similarities[query_idx] = -float('inf')
                labels[query_idx] = 0

                # Check if there are any positive examples
                if labels.sum() == 0:
                    continue

                # Sort all items by similarity
                sorted_indices = torch.argsort(similarities, descending=True)
                sorted_labels = labels[sorted_indices]

                # Calculate average precision at k
                ap = self.calculate_average_precision(sorted_labels.numpy()[:k])
                if ap > 0:  # Only include if there are positive examples
                    average_precisions.append(ap)

            map_scores[f'mAP@{k}'] = np.mean(average_precisions) if average_precisions else 0.0

        return map_scores

    def calculate_average_precision(self, labels):
        """Calculate Average Precision for a single query"""
        num_relevant = np.sum(labels)

        if num_relevant == 0:
            return 0.0

        precision_at_i = []
        num_relevant_so_far = 0

        for i, label in enumerate(labels):
            if label == 1:
                num_relevant_so_far += 1
                precision_at_i.append(num_relevant_so_far / (i + 1))

        return np.mean(precision_at_i) if precision_at_i else 0.0

    def evaluate_triplet_loss(self, embeddings, margin=1.0):
        """Evaluate triplet loss on embeddings"""
        manifold = PoincareManifold()
        triplet_losses = []
        valid_triplets = 0

        # Generate triplets: (anchor, positive, negative)
        for anchor_idx in range(self.num_nodes):
            anchor_emb = embeddings[anchor_idx:anchor_idx+1]

            # Find positive examples
            positives = [idx for idx in range(self.num_nodes)
                        if (anchor_idx, idx) in self.positive_pairs]

            # Find negative examples
            negatives = [idx for idx in range(self.num_nodes)
                        if (anchor_idx, idx) in self.negative_pairs]

            if not positives or not negatives:
                continue

            # Sample some triplets for this anchor
            for pos_idx in positives[:2]:  # Limit to avoid too many triplets
                pos_emb = embeddings[pos_idx:pos_idx+1]

                for neg_idx in negatives[:3]:  # Limit negatives
                    neg_emb = embeddings[neg_idx:neg_idx+1]

                    # Compute distances
                    pos_dist = manifold.distance(anchor_emb, pos_emb)
                    neg_dist = manifold.distance(anchor_emb, neg_emb)

                    # Compute triplet loss
                    loss = F.relu(margin + pos_dist - neg_dist)
                    triplet_losses.append(loss.item())
                    valid_triplets += 1

        avg_triplet_loss = np.mean(triplet_losses) if triplet_losses else 0.0
        return avg_triplet_loss, valid_triplets

# ========== Step 3: Enhanced Training with Evaluation ==========
def train_model(
    dim=10,  # Increased dimension for better capacity
    lr=0.1,  # Increased learning rate
    epochs=200,  # More epochs
    negs=20,  # More negatives for better contrast
    batch_size=8,  # Use all edges at once since we have small dataset
    eval_each=20,
    sparse=False,
    margin=1.0,  # Larger margin for better separation
    evaluate_metrics=True  # NEW: Enable evaluation during training
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    create_mammal_checkpoint()
    edges, node_to_idx = load_edges('filtered_mammals.csv')

    print(f"Dataset loaded: {len(node_to_idx)} nodes, {len(edges)} edges")

    # Initialize model and optimizer with better stability
    model = HyperbolicEmbedding(num_nodes=len(node_to_idx), dim=dim, sparse=sparse).to(device)
    optimizer = RiemannianSGD(model.parameters(), lr=lr)

    # NEW: Initialize evaluator
    evaluator = EmbeddingEvaluator(node_to_idx) if evaluate_metrics else None

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print("Initial embedding norms:", torch.norm(model.get_all_embeddings(), dim=1).cpu().numpy())

    dataset = GraphDataset(edges, len(node_to_idx), batch_size=batch_size, num_negs=negs)

    print("\nTraining started...\n")

    # Storage for metrics over time
    metrics_history = {
        'epochs': [],
        'losses': [],
        'mAP@1': [],
        'mAP@3': [],
        'mAP@5': [],
        'triplet_loss': [],
        'mean_average_precision': []
    }

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for pos, neg in dataset:
            pos, neg = pos.to(device), neg.to(device)

            # Get embeddings
            pos_emb = model(pos)
            neg_emb = model(neg)

            # Compute distances
            pos_dist = model.distance(pos_emb[:, 0], pos_emb[:, 1])
            neg_dist = model.distance(neg_emb[:, 0], neg_emb[:, 1])

            # Check for NaN values
            if torch.isnan(pos_dist).any() or torch.isnan(neg_dist).any():
                print(f"NaN detected at epoch {epoch+1}, batch {num_batches+1}")
                break

            # Expand positive distances to match negative batch size
            pos_dist_expanded = pos_dist.repeat_interleave(negs)

            # Only use the first len(pos_dist_expanded) negative distances
            neg_dist_used = neg_dist[:len(pos_dist_expanded)]

            # Compute loss
            loss = hinge_loss(pos_dist_expanded, neg_dist_used, margin)

            # Check for NaN loss
            if torch.isnan(loss):
                print(f"NaN loss at epoch {epoch+1}, batch {num_batches+1}")
                break

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Normalize embeddings to Poincaré ball
            with torch.no_grad():
                model.lt.weight.data = model.manifold.normalize(model.lt.weight.data)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0

        # NEW: Evaluate metrics periodically
        if (epoch % eval_each == 0 or epoch == epochs - 1) and evaluate_metrics:
            print(f"Epoch {epoch+1:03d}: Loss = {avg_loss:.6f}")

            # Get current embeddings for evaluation
            current_embeddings = model.get_all_embeddings()

            # Check embedding health
            norms = torch.norm(current_embeddings, dim=1)
            print(f"  Embedding norms: min={norms.min():.6f}, max={norms.max():.6f}, mean={norms.mean():.6f}")

            # Evaluate mAP
            map_scores = evaluator.evaluate_map(current_embeddings, k_values=[1, 3, 5])

            # Evaluate triplet loss
            triplet_loss, num_triplets = evaluator.evaluate_triplet_loss(current_embeddings, margin=margin)

            # Calculate overall mean average precision
            overall_map = np.mean(list(map_scores.values()))

            # Print metrics
            print(f"  mAP@1: {map_scores['mAP@1']:.4f}, mAP@3: {map_scores['mAP@3']:.4f}, mAP@5: {map_scores['mAP@5']:.4f}")
            print(f"  Mean Average Precision: {overall_map:.4f}")
            print(f"  Triplet Loss: {triplet_loss:.4f} (from {num_triplets} triplets)")

            # Store metrics
            metrics_history['epochs'].append(epoch + 1)
            metrics_history['losses'].append(avg_loss)
            metrics_history['mAP@1'].append(map_scores['mAP@1'])
            metrics_history['mAP@3'].append(map_scores['mAP@3'])
            metrics_history['mAP@5'].append(map_scores['mAP@5'])
            metrics_history['triplet_loss'].append(triplet_loss)
            metrics_history['mean_average_precision'].append(overall_map)

            # Show sample distances
            sample_dist = model.distance(current_embeddings[0:1], current_embeddings[1:2])
            print(f"  Sample distance: {sample_dist.item():.6f}")

            if torch.isnan(norms).any():
                print("WARNING: NaN in embeddings detected!")
                break

        elif epoch % eval_each == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:03d}: Loss = {avg_loss:.6f}")
            with torch.no_grad():
                all_emb = model.get_all_embeddings()
                norms = torch.norm(all_emb, dim=1)
                if epoch % (eval_each * 2) == 0:
                    print(f"  Embedding norms: min={norms.min():.6f}, max={norms.max():.6f}, mean={norms.mean():.6f}")
                    sample_dist = model.distance(all_emb[0:1], all_emb[1:2])
                    print(f"  Sample distance: {sample_dist.item():.6f}")

    # Final comprehensive evaluation
    if evaluate_metrics:
        print("\n" + "="*60)
        print("=== FINAL EVALUATION RESULTS ===")
        print("="*60)

        final_embeddings = model.get_all_embeddings()

        # Final mAP scores
        final_map_scores = evaluator.evaluate_map(final_embeddings, k_values=[1, 2, 3, 5])
        final_triplet_loss, final_num_triplets = evaluator.evaluate_triplet_loss(final_embeddings, margin=margin)
        final_overall_map = np.mean(list(final_map_scores.values()))

        print(f"\nFinal Metrics:")
        for metric, score in final_map_scores.items():
            print(f"  {metric}: {score:.4f}")
        print(f"  Overall Mean Average Precision: {final_overall_map:.4f}")
        print(f"  Final Triplet Loss: {final_triplet_loss:.4f}")

        # Performance interpretation
        print(f"\nPerformance Analysis:")
        if final_map_scores['mAP@1'] > 0.5:
            print(f"  ✓ Good nearest neighbor precision (mAP@1 = {final_map_scores['mAP@1']:.4f})")
        else:
            print(f"  ⚠ Low nearest neighbor precision (mAP@1 = {final_map_scores['mAP@1']:.4f})")

        if final_triplet_loss < 0.5:
            print(f"  ✓ Good embedding separation (Triplet Loss = {final_triplet_loss:.4f})")
        else:
            print(f"  ⚠ Poor embedding separation (Triplet Loss = {final_triplet_loss:.4f})")

        if final_overall_map > 0.4:
            print(f"  ✓ Decent overall retrieval quality (Overall mAP = {final_overall_map:.4f})")
        else:
            print(f"  ⚠ Low overall retrieval quality (Overall mAP = {final_overall_map:.4f})")

    # Nearest neighbor analysis (unchanged)
    print("\n=== NEAREST NEIGHBOR ANALYSIS ===")
    idx_to_node = {v: k for k, v in node_to_idx.items()}

    # Get embedding statistics
    all_emb = model.get_all_embeddings()
    norms = torch.norm(all_emb, dim=1)
    print(f"Embedding norms - min: {norms.min():.4f}, max: {norms.max():.4f}, mean: {norms.mean():.4f}")

    # Show nearest neighbors for first few nodes
    for node_name in list(node_to_idx.keys())[:3]:
        node_idx = node_to_idx[node_name]
        emb = model.get_embedding(node_idx)
        dists = model.distance(emb, all_emb).cpu().numpy()
        nearest = np.argsort(dists)[1:6]  # Skip self, show top 5

        print(f"\nNearest to '{node_name}':")
        for idx in nearest:
            print(f"  {idx_to_node[idx]} - dist: {dists[idx]:.4f}")

    # Save enhanced checkpoint
    checkpoint = {
        'embeddings': model.get_all_embeddings(),
        'node_to_idx': node_to_idx,
        'idx_to_node': idx_to_node,
        'model_state_dict': model.state_dict(),
        'hyperparameters': {
            'dim': dim,
            'lr': lr,
            'epochs': epochs,
            'negs': negs,
            'batch_size': batch_size,
            'margin': margin
        },
        'metrics_history': metrics_history if evaluate_metrics else None,
        'final_metrics': {
            'mAP_scores': final_map_scores,
            'overall_mAP': final_overall_map,
            'triplet_loss': final_triplet_loss
        } if evaluate_metrics else None
    }
    torch.save(checkpoint, 'mammals_poincare_embeddings.pth')
    print("\n✓ Enhanced checkpoint saved as 'mammals_poincare_embeddings.pth'")

    return metrics_history if evaluate_metrics else None

# ========== Step 4: Run Enhanced Training ==========
if __name__ == '__main__':
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run training with evaluation metrics enabled
    print("Starting enhanced training with evaluation metrics...")
    metrics_history = train_model(
        dim=10,
        lr=0.1,
        epochs=200,
        negs=20,
        batch_size=8,
        eval_each=40,  # Evaluate every 40 epochs to reduce output
        margin=1.0,
        evaluate_metrics=True  # Enable metric evaluation
    )

# new cell

import pandas as pd
import os

csv_path = "/content/poincare-embeddings/wordnet/mammals_embedding.csv"

if os.path.exists(csv_path):
    print(f"Inspecting the first 5 rows of {csv_path}:")
    try:
        df_inspect = pd.read_csv(csv_path, dtype={"noun_id": str, "embedding": str}, low_memory=False)
        print(df_inspect.head())

        # Also print the raw string format of the first few embeddings
        print("\nRaw embedding string format for the first 5 entries:")
        for i in range(min(5, len(df_inspect))):
            print(f"Entry {i}: {df_inspect.loc[i, 'embedding']}")

    except Exception as e:
        print(f"Error reading CSV with pandas: {e}")
else:
    print(f"CSV file not found: {csv_path}")
    print("Please ensure the file exists at this path.")

