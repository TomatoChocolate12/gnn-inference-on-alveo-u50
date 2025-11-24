import torch
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

# Model
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, 16)  # First layer: 1433 -> 16
        self.conv2 = GCNConv(16, num_classes)    # Second layer (not used by HLS)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Load Cora dataset
dataset = Planetoid(root='data', name='Cora')
data = dataset[0]

# Train model (your training code here)
model = GCN(dataset.num_node_features, dataset.num_classes)
# ... training loop ...

# Save model
torch.save(model, 'models/gcn_model.pth')