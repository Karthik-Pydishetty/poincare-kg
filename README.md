# Poincaré Knowledge Graph

A Python implementation of Poincaré embeddings for knowledge graph representation learning in hyperbolic space.

## Overview

This project implements Poincaré embeddings for knowledge graphs, which represent hierarchical relationships in hyperbolic space. The implementation includes:

- **Poincaré Ball Manifold**: Hyperbolic geometry operations
- **Energy Functions**: Distance-based and entailment cone energy functions
- **Graph Processing**: Tools for loading and processing graph data
- **Training Pipeline**: Complete training and evaluation framework
- **Evaluation Metrics**: MAP, triplet loss, and reconstruction metrics

## Files

- `Embed.py` - Main embedding training and evaluation code
- `Energy_function.py` - Energy function implementations for hyperbolic embeddings
- `Graph.py` - Graph data loading and processing utilities
- `Graph_dataset.py` - Dataset classes for graph training
- `Poincare.py` - Poincaré manifold implementation
- `Reconstruction.py` - Graph reconstruction evaluation (placeholder)

## Setup

### 1. Create Virtual Environment

```bash
python3 -m venv poincare_env
```

### 2. Activate Virtual Environment

```bash
source poincare_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Dependencies

- `torch>=1.9.0` - PyTorch for deep learning
- `numpy>=1.19.0` - Numerical computing
- `pandas>=1.3.0` - Data manipulation
- `scikit-learn>=1.0.0` - Machine learning utilities
- `h5py>=3.1.0` - HDF5 file format support
- `tqdm>=4.62.0` - Progress bars
- `iopath>=0.1.0` - File I/O utilities

## Usage

### Basic Training

```python
from Embed import train_model

# Train Poincaré embeddings
train_model(
    dim=10,
    lr=0.1,
    epochs=200,
    negs=20,
    batch_size=8,
    margin=1.0
)
```

### Loading and Processing Data

```python
from Graph import load_adjacency_matrix
from Graph_dataset import VectorBatchedDataset

# Load graph data
adj_data = load_adjacency_matrix('path/to/data.h5', format='hdf5')

# Create dataset
dataset = VectorBatchedDataset(
    idx=idx_array,
    embeddings=emb_vectors,
    weights=weights,
    nnegs=50,
    batch_size=128
)
```

## Features

- **Hyperbolic Geometry**: Full implementation of Poincaré ball manifold
- **Multiple Energy Functions**: Distance-based and entailment cone energy functions
- **Evaluation Metrics**: MAP, triplet loss, and reconstruction accuracy
- **Flexible Data Loading**: Support for HDF5 and CSV formats
- **Batch Processing**: Efficient training with negative sampling

## Research Context

This implementation is based on research in hyperbolic embeddings for knowledge graphs, particularly useful for representing hierarchical relationships that are difficult to capture in Euclidean space.

## License

This project is for research purposes. Please ensure compliance with any applicable licenses for the underlying research papers and datasets used. 