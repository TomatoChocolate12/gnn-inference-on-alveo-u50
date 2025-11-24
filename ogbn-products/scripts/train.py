#!/usr/bin/env python3
"""
Train a GCN model for ogbn-products dataset.

This script trains a single-layer GCN model matching the HLS kernel
dimensions (num_features -> num_classes for ogbn-products).

Usage:
    python scripts/train.py --output models/gcn_model.pth
"""

import argparse
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from ogb.nodeproppred import PygNodePropPredDataset

# Model - Single layer: num_features -> num_classes
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        # Single layer: num_features -> num_classes (matches HLS kernel)
        self.conv1 = GCNConv(num_features, num_classes)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return x


def train(model, data, split_idx, optimizer, epochs=100):
    """Training loop."""
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        # Handle ogbn-products label format (may be 2D)
        y = data.y.squeeze() if data.y.dim() > 1 else data.y
        loss = F.cross_entropy(out[split_idx['train']], y[split_idx['train']])
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='models/gcn_model.pth',
                       help='Path to save trained model')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    args = parser.parse_args()
    
    # Load ogbn-products dataset
    print("Loading ogbn-products dataset...")
    dataset = PygNodePropPredDataset(name='ogbn-products', root='data')
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.edge_index.shape[1]}")
    print(f"  Features: {data.x.shape[1]}")
    print(f"  Classes: {dataset.num_classes}")
    
    # Verify dimensions match HLS kernel
    if data.x.shape[1] != 100:
        raise ValueError(f"Expected 100 input features, got {data.x.shape[1]}")
    
    # Create model - single layer
    print(f"\nCreating single-layer model: {data.x.shape[1]} -> {dataset.num_classes}")
    model = GCN(data.x.shape[1], dataset.num_classes)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Train model
    print(f"\nTraining for {args.epochs} epochs...")
    train(model, data, split_idx, optimizer, epochs=args.epochs)
    
    # Save model
    print(f"\nSaving model to {args.output}")
    import pathlib
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, args.output)
    print("✓ Model saved successfully!")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())