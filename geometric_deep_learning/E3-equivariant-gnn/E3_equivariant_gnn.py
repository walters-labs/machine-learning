# Example: PyTorch Geometric
from torch_geometric.datasets import QM9
dataset = QM9(root='data/QM9')

from pymatgen.core import Structure
from ase import Atoms
import networkx as nx
import itertools

# Read a CIF from COD
structure = Structure.from_file("data/CIF/1100118.cif")

# Build a simple graph
G = nx.Graph()
for i, site in enumerate(structure.sites):
    G.add_node(i, element=site.specie.symbol, coords=site.coords)

# Add edges for pairs within cutoff distance
cutoff = 3.0
for i, site in enumerate(structure):
    neighbors = structure.get_neighbors(site, r=cutoff)
    for neighbor in neighbors:
        j = neighbor.index
        if i < j:
            rel_vec = neighbor.coords - site.coords
            G.add_edge(
                i,
                j,
                weight=neighbor.nn_distance,
                rel_vec=rel_vec
            )

### ------- Convert to PyTorch Geometric Data -------- ###

import torch
from torch_geometric.data import Data

# Node features: atomic numbers
x = torch.tensor(
    [site.specie.number for site in structure],
    dtype=torch.long
).unsqueeze(-1)

# Positions
pos = torch.tensor(
    [site.coords for site in structure],
    dtype=torch.float
)

# Edges
edge_index = []
edge_attr = []

for i, j, data in G.edges(data=True):
    edge_index.append([i, j])
    edge_index.append([j, i])  # undirected graph

    edge_attr.append(data["rel_vec"])
    edge_attr.append(-data["rel_vec"])

edge_index = torch.tensor(edge_index).t().contiguous()
edge_attr = torch.tensor(edge_attr, dtype=torch.float)

data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)

### ------- Minimal EGNN Layer Implementation -------- ###

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

class MinimalEGNNLayer(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super().__init__()
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * in_features + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        self.node_mlp = nn.Sequential(
            nn.Linear(in_features + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, in_features)
        )
        
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, h, x, edge_index):
        row, col = edge_index

        # Relative positions
        rel = x[row] - x[col]
        dist2 = (rel ** 2).sum(dim=1, keepdim=True)

        # Edge features
        edge_input = torch.cat([h[row], h[col], dist2], dim=1)
        m_ij = self.edge_mlp(edge_input)

        # Node update
        m_i = scatter(m_ij, row, dim=0, dim_size=h.size(0), reduce='add')
        h = self.node_mlp(torch.cat([h, m_i], dim=1))

        # Coordinate update (equivariant!)
        coord_update = rel * self.coord_mlp(m_ij)
        delta_x = scatter(coord_update, row, dim=0, dim_size=x.size(0), reduce='add')
        x = x + delta_x

        return h, x

### ------- EGNN Model --------- ###

class EGNN(nn.Module):
    def __init__(self, in_features, hidden_dim, num_layers, out_dim):
        super().__init__()
        self.input_proj = nn.Linear(in_features, hidden_dim)  # 11 -> hidden_dim
        self.layers = nn.ModuleList([
            MinimalEGNNLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, h, x, edge_index, batch):
        h = self.input_proj(h)  # h is already float [N, 11]
        for layer in self.layers:
            h, x = layer(h, x, edge_index)
        h_graph = scatter(h, batch, dim=0, reduce='mean')
        return self.readout(h_graph)

# Instantiate with QM9's 11 node features
model = EGNN(in_features=11, hidden_dim=128, num_layers=4, out_dim=1)

### -------- Training loop (simplified) -------- ###

from torch_geometric.loader import DataLoader

loader = DataLoader(dataset[:1000], batch_size=32, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for batch in loader:
    pred = model(batch.x, batch.pos, batch.edge_index, batch.batch)
    loss = nn.MSELoss()(pred.squeeze(), batch.y[:, 7])  # e.g. target 7 = U0
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

### -------- Full training loop with validation and learning rate scheduling ------- ###

from torch_geometric.loader import DataLoader
import numpy as np

# ── Data splits ──────────────────────────────────────────────────────────────
torch.manual_seed(42)
perm = torch.randperm(len(dataset))
train_data = dataset[perm[:8000]]
val_data   = dataset[perm[8000:9000]]
test_data  = dataset[perm[9000:10000]]

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=32, shuffle=False)

# ── Normalize target (U0 is target index 7) ──────────────────────────────────
TARGET = 7
y_train = torch.cat([d.y[:, TARGET] for d in train_data])
y_mean  = y_train.mean().item()
y_std   = y_train.std().item()

def normalize(y):   return (y - y_mean) / y_std
def denormalize(y): return y * y_std + y_mean

# ── Model + optimizer ─────────────────────────────────────────────────────────
model = EGNN(in_features=11, hidden_dim=128, num_layers=4, out_dim=1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.5
)

# ── Training loop ─────────────────────────────────────────────────────────────
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, total_mae, n = 0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            pred = model(batch.x, batch.pos, batch.edge_index, batch.batch).squeeze()
            target = normalize(batch.y[:, TARGET])

            loss = nn.MSELoss()(pred, target)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # MAE in original units
            mae = (denormalize(pred) - batch.y[:, TARGET]).abs().mean().item()
            total_loss += loss.item() * batch.num_graphs
            total_mae  += mae         * batch.num_graphs
            n          += batch.num_graphs

    return total_loss / n, total_mae / n

for epoch in range(1, 51):
    train_loss, train_mae = run_epoch(train_loader, train=True)
    val_loss,   val_mae   = run_epoch(val_loader,   train=False)
    scheduler.step(val_loss)

    if epoch % 5 == 0:
        print(f"Epoch {epoch:3d} | "
              f"Train MSE: {train_loss:.4f}  MAE: {train_mae:.4f} eV | "
              f"Val   MSE: {val_loss:.4f}  MAE: {val_mae:.4f} eV")