"""
Equivariant Graph Neural Network (EGNN) — Improved Implementation
Fixes applied:
  1. Proper atomic number embedding (nn.Embedding) instead of casting longs to float
  2. Unit-vector coord updates + mean aggregation to prevent coordinate drift/explosion
  3. Distance (not squared distance) as edge feature — more numerically stable
  4. Residual connections in node MLP — critical for deeper networks
  5. Periodic boundary conditions via pymatgen CrystalNN for crystal structures
  6. Device handling (GPU/CPU)
  7. Cleaner gradient context via torch.set_grad_enabled
  8. Consistent dtype handling throughout
"""

# ── Imports ───────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import QM9
from torch_geometric.utils import scatter

# ── Device ────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ════════════════════════════════════════════════════════════════════════════════
# CIF → PyG Graph  (crystal structure preprocessing)
# ════════════════════════════════════════════════════════════════════════════════

def cif_to_pyg(cif_path: str, cutoff: float = 5.0) -> Data:
    """
    Convert a CIF file to a PyTorch Geometric Data object.

    Key fix: uses CrystalNN (periodic-boundary-aware) instead of a naive
    distance cutoff that misses atoms across cell boundaries.
    """
    from pymatgen.core import Structure
    from pymatgen.analysis.graphs import StructureGraph
    from pymatgen.analysis.local_env import CrystalNN
    import networkx as nx

    structure = Structure.from_file(cif_path)

    # ── Periodic-boundary-correct neighbor graph ──────────────────────────────
    sg = StructureGraph.with_local_env_strategy(structure, CrystalNN())
    G = sg.graph.to_undirected()

    # ── Node features: atomic number (long, for nn.Embedding) ─────────────────
    # FIX: keep as long — we'll use nn.Embedding, not nn.Linear, for node input
    atomic_numbers = torch.tensor(
        [site.specie.number for site in structure],
        dtype=torch.long,
    )  # shape [N]

    # ── Positions ─────────────────────────────────────────────────────────────
    pos = torch.tensor(
        [site.coords for site in structure],
        dtype=torch.float,
    )  # shape [N, 3]

    # ── Edges ─────────────────────────────────────────────────────────────────
    edge_index, edge_attr = [], []
    for i, j, edata in G.edges(data=True):
        # image offset for periodic images
        offset = edata.get("to_jimage", (0, 0, 0))
        offset_cart = structure.lattice.get_cartesian_coords(offset)
        rel_vec = (structure[j].coords + offset_cart) - structure[i].coords

        edge_index += [[i, j], [j, i]]
        edge_attr  += [rel_vec, -rel_vec]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(edge_attr,  dtype=torch.float)

    return Data(x=atomic_numbers, pos=pos, edge_index=edge_index, edge_attr=edge_attr)


# ════════════════════════════════════════════════════════════════════════════════
# EGNN Layer
# ════════════════════════════════════════════════════════════════════════════════

class EGNNLayer(nn.Module):
    """
    Single E(3)-equivariant message-passing layer.

    Why E(3)-equivariant?
    ─────────────────────
    • Edge messages use ‖rᵢⱼ‖ (a scalar) → rotation-invariant.
    • Coordinate update = (unit vector rᵢⱼ/‖rᵢⱼ‖) × (learned scalar φ) →
      rotates/reflects with the coordinate frame, i.e., equivariant.
    • Translations cancel in relative positions rᵢⱼ = xᵢ − xⱼ.

    Fixes vs. original:
    ───────────────────
    • dist (not dist²) as edge feature — better-conditioned gradients.
    • Unit-vector normalization on coord update — prevents explosion.
    • Mean aggregation for coord update — zero net force in expectation.
    • Residual connection on node features — stable deep networks.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        # 2 * hidden_dim (node feats i & j) + 1 (distance scalar)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # hidden_dim (node feat) + hidden_dim (aggregated messages)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # scalar weight for coord update direction
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h: torch.Tensor,          # [N, hidden_dim]  node features
        x: torch.Tensor,          # [N, 3]           node coordinates
        edge_index: torch.Tensor, # [2, E]
    ):
        row, col = edge_index  # row = source, col = target

        # ── Relative positions & distances ────────────────────────────────────
        rel  = x[row] - x[col]                              # [E, 3]
        dist = rel.norm(dim=1, keepdim=True).clamp(min=1e-8)  # [E, 1]  FIX: use dist, not dist²
        unit = rel / dist                                    # [E, 3]  unit vector

        # ── Edge messages ─────────────────────────────────────────────────────
        m_ij = self.edge_mlp(torch.cat([h[row], h[col], dist], dim=1))  # [E, hidden]

        # ── Node update (with residual) ────────────────────────────────────────
        agg  = scatter(m_ij, row, dim=0, dim_size=h.size(0), reduce="sum")  # [N, hidden]
        h_new = self.node_mlp(torch.cat([h, agg], dim=1))
        h = h + h_new  # FIX: residual connection

        # ── Coordinate update (equivariant) ───────────────────────────────────
        # unit_vec * scalar  →  equivariant vector, mean-aggregated → no drift
        coord_weight  = self.coord_mlp(m_ij)                           # [E, 1]
        coord_update  = unit * coord_weight                            # [E, 3]
        delta_x = scatter(coord_update, row, dim=0, dim_size=x.size(0), reduce="mean")  # FIX: mean
        x = x + delta_x

        return h, x


# ════════════════════════════════════════════════════════════════════════════════
# EGNN Model
# ════════════════════════════════════════════════════════════════════════════════

class EGNN(nn.Module):
    """
    Full EGNN: embedding → stacked EGNNLayers → global mean-pool → MLP readout.

    FIX: uses nn.Embedding for atomic numbers (correct for integer atom types)
         instead of nn.Linear on a float cast of a long tensor.
    """

    def __init__(
        self,
        num_atom_types: int = 100,  # max atomic number + 1
        hidden_dim:     int = 128,
        num_layers:     int = 4,
        out_dim:        int = 1,
    ):
        super().__init__()

        # FIX: Embedding handles integer atom-type indices properly
        self.embedding = nn.Embedding(num_atom_types, hidden_dim)

        self.layers = nn.ModuleList(
            [EGNNLayer(hidden_dim) for _ in range(num_layers)]
        )

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        x:          torch.Tensor,  # [N]     long — atomic numbers
        pos:        torch.Tensor,  # [N, 3]  float — coordinates
        edge_index: torch.Tensor,  # [2, E]
        batch:      torch.Tensor,  # [N]     long — graph assignment
    ) -> torch.Tensor:

        h = self.embedding(x.squeeze(-1))  # [N, hidden_dim]  FIX: Embedding, not Linear

        for layer in self.layers:
            h, pos = layer(h, pos, edge_index)

        # Global mean-pool over nodes in each graph
        h_graph = scatter(h, batch, dim=0, reduce="mean")   # [B, hidden_dim]
        return self.readout(h_graph)                         # [B, out_dim]


# ════════════════════════════════════════════════════════════════════════════════
# QM9 Training
# ════════════════════════════════════════════════════════════════════════════════

def build_edge_index_from_pos(pos: torch.Tensor, cutoff: float = 5.0):
    """Build radius-graph edges for a single molecule (used if QM9 edge_index absent)."""
    n = pos.size(0)
    rows, cols = [], []
    for i in range(n):
        for j in range(n):
            if i != j and (pos[i] - pos[j]).norm() < cutoff:
                rows.append(i)
                cols.append(j)
    return torch.tensor([rows, cols], dtype=torch.long)


def train_qm9(
    data_root:   str = "data/QM9",
    target_idx:  int = 7,       # U0 (internal energy at 0K)
    hidden_dim:  int = 128,
    num_layers:  int = 4,
    num_epochs:  int = 50,
    batch_size:  int = 32,
    lr:          float = 1e-3,
    train_size:  int = 8000,
    val_size:    int = 1000,
):
    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = QM9(root=data_root)

    torch.manual_seed(42)
    perm       = torch.randperm(len(dataset))
    train_data = dataset[perm[:train_size]]
    val_data   = dataset[perm[train_size : train_size + val_size]]

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False)

    # ── Target normalisation ──────────────────────────────────────────────────
    y_train = torch.cat([d.y[:, target_idx] for d in train_data])
    y_mean, y_std = y_train.mean().item(), y_train.std().item()

    def normalize(y):   return (y - y_mean) / y_std
    def denormalize(y): return y * y_std + y_mean

    # ── Model ─────────────────────────────────────────────────────────────────
    # QM9 node features are one-hot + extras (11 dims); we use atomic number
    # which is available as batch.z (long tensor) in PyG's QM9.
    model     = EGNN(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    # ── Epoch function ────────────────────────────────────────────────────────
    def run_epoch(loader, train: bool):
        model.train(train)
        total_loss = total_mae = n = 0

        with torch.set_grad_enabled(train):   # FIX: cleaner than enable_grad/no_grad ctx
            for batch in loader:
                batch = batch.to(device)      # FIX: move data to device

                # QM9 provides batch.z (atomic numbers, long) and batch.pos
                pred   = model(batch.z, batch.pos, batch.edge_index, batch.batch).squeeze()
                target = normalize(batch.y[:, target_idx])

                loss = nn.functional.mse_loss(pred, target)

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                mae = (denormalize(pred.detach()) - batch.y[:, target_idx]).abs().mean().item()
                bs  = batch.num_graphs
                total_loss += loss.item() * bs
                total_mae  += mae         * bs
                n          += bs

        return total_loss / n, total_mae / n

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, num_epochs + 1):
        train_loss, train_mae = run_epoch(train_loader, train=True)
        val_loss,   val_mae   = run_epoch(val_loader,   train=False)
        scheduler.step(val_loss)

        if epoch % 5 == 0:
            print(
                f"Epoch {epoch:3d} | "
                f"Train  MSE: {train_loss:.4f}  MAE: {train_mae:.4f} eV | "
                f"Val    MSE: {val_loss:.4f}  MAE: {val_mae:.4f} eV"
            )

    return model


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    trained_model = train_qm9()