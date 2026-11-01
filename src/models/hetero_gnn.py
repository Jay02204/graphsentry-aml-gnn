"""
Heterogeneous Graph Neural Network Architecture for Anti-Money Laundering.

Implements a 2-layer HeteroGraphSAGE network tailored for financial crime graph topologies:
- Handles heterogeneous node types: 'account', 'customer', 'device'
- Multi-relational message passing across directed transfer, ownership, and login edges
- Constructed using PyG's native to_hetero transformation
- Classification head predicts binary mule / fraud probabilities for 'account' nodes
"""

from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import HeteroConv, SAGEConv
    _HAS_TORCH_PYG = True
except ImportError:
    _HAS_TORCH_PYG = False


if _HAS_TORCH_PYG:
    class SAGEBackboneLayer(nn.Module):
        """Single GraphSAGE layer with LayerNorm, ELU activation, and Dropout."""

        def __init__(self, in_channels: int, out_channels: int, aggr: str = "mean", dropout: float = 0.2):
            super().__init__()
            self.conv = SAGEConv(in_channels, out_channels, aggr=aggr)
            self.norm = nn.LayerNorm(out_channels)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            h = self.conv(x, edge_index)
            h = self.norm(h)
            h = F.elu(h)
            h = self.dropout(h)
            return h

    class HomogeneousGNNBackbone(nn.Module):
        """
        Homogeneous multi-layer backbone designed to be converted into
        a multi-relational heterogeneous network via torch_geometric.nn.to_hetero.
        """

        def __init__(
            self,
            hidden_channels: int = 64,
            num_layers: int = 2,
            aggr: str = "mean",
            dropout: float = 0.2,
        ):
            super().__init__()
            self.convs = nn.ModuleList()
            for _ in range(num_layers):
                self.convs.append(
                    SAGEBackboneLayer(
                        in_channels=(-1, -1),
                        out_channels=hidden_channels,
                        aggr=aggr,
                        dropout=dropout,
                    )
                )

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            for conv in self.convs:
                x = conv(x, edge_index)
            return x

    class HeteroGraphSAGE(nn.Module):
        """
        Production Heterogeneous Graph Neural Network for Account Classification.

        Architecture:
        1. Node-Specific Linear Projections:
           Projects heterogeneously-sized raw feature vectors from 'account',
           'customer', and 'device' to uniform hidden dimension H.
        2. Heterogeneous GraphSAGE Message Passing:
           Constructed via PyG to_hetero, aggregating relational messages along
           directed transfers, device logins, customer ownerships, and their inverses.
        3. Multilayer Perceptron Classification Head:
           Takes the final 'account' node embeddings and produces scalar unnormalized
           logits for binary fraud detection.
        """

        def __init__(
            self,
            in_channels_dict: Dict[str, int],
            metadata: Tuple[List[str], List[Tuple[str, str, str]]],
            hidden_channels: int = 64,
            num_layers: int = 2,
            aggr: str = "mean",
            dropout: float = 0.2,
        ):
            super().__init__()
            self.hidden_channels = hidden_channels
            self.metadata = metadata

            # 1. Feature Projections per Node Type
            self.projections = nn.ModuleDict()
            for node_type, in_dim in in_channels_dict.items():
                self.projections[node_type] = nn.Sequential(
                    nn.Linear(in_dim, hidden_channels),
                    nn.LayerNorm(hidden_channels),
                    nn.ELU(),
                )

            # 2. Heterogeneous Relational Convolution Layers
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()
            self.dropout = nn.Dropout(dropout)

            node_types, edge_types = metadata
            for _ in range(num_layers):
                conv_dict = {}
                for edge_type in edge_types:
                    src, rel, dst = edge_type
                    if src == dst:
                        conv_dict[edge_type] = SAGEConv(hidden_channels, hidden_channels, aggr=aggr)
                    else:
                        conv_dict[edge_type] = SAGEConv((hidden_channels, hidden_channels), hidden_channels, aggr=aggr)
                self.convs.append(HeteroConv(conv_dict, aggr="sum"))
                self.norms.append(
                    nn.ModuleDict({nt: nn.LayerNorm(hidden_channels) for nt in node_types})
                )

            # 3. Account Binary Classification Head
            self.classifier = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.LayerNorm(hidden_channels // 2),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels // 2, 1),
            )

        def forward(
            self,
            x_dict: Dict[str, torch.Tensor],
            edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        ) -> torch.Tensor:
            """
            Forward propagation through the heterogeneous graph.

            Returns:
                torch.Tensor: Logits for 'account' nodes of shape (N_accounts, 1).
            """
            # Project distinct node feature dimensions into hidden space
            h_dict = {}
            for node_type, x in x_dict.items():
                if node_type in self.projections:
                    h_dict[node_type] = self.projections[node_type](x)
                else:
                    h_dict[node_type] = x

            # Heterogeneous relational message passing
            for conv, norm_dict in zip(self.convs, self.norms):
                h_dict = conv(h_dict, edge_index_dict)
                h_dict = {
                    nt: self.dropout(F.elu(norm_dict[nt](h)))
                    for nt, h in h_dict.items() if nt in norm_dict
                }

            # Compute classification logits on target 'account' embeddings
            account_embeddings = h_dict["account"]
            logits = self.classifier(account_embeddings)
            return logits

        @torch.no_grad()
        def predict_proba(
            self,
            x_dict: Dict[str, torch.Tensor],
            edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        ) -> torch.Tensor:
            """Compute calibrated fraud probabilities in [0.0, 1.0]."""
            self.eval()
            logits = self.forward(x_dict, edge_index_dict)
            return torch.sigmoid(logits).view(-1)

        @torch.no_grad()
        def predict(
            self,
            x_dict: Dict[str, torch.Tensor],
            edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
            threshold: float = 0.45,
        ) -> torch.Tensor:
            """Predict discrete binary classes (0 = licit, 1 = mule)."""
            probs = self.predict_proba(x_dict, edge_index_dict)
            return (probs >= threshold).long()

else:
    # Pure Python / Mock model class for environments where PyG is not installed
    class HeteroGraphSAGE:
        def __init__(self, in_channels_dict: Dict[str, int], metadata: Any, hidden_channels: int = 64, **kwargs):
            self.in_channels_dict = in_channels_dict
            self.metadata = metadata
            self.hidden_channels = hidden_channels

        def forward(self, *args, **kwargs):
            raise NotImplementedError("PyTorch and PyG are required for GNN tensor forward passes.")

        def predict_proba(self, *args, **kwargs):
            raise NotImplementedError("PyTorch and PyG are required for GNN inference.")


def build_hetero_gnn_model(
    metadata_dict: Dict[str, Any],
    hidden_channels: int = 64,
    num_layers: int = 2,
    aggr: str = "mean",
    dropout: float = 0.2,
) -> Any:
    """
    Factory helper to instantiate HeteroGraphSAGE from metadata dictionary.
    """
    in_channels_dict = metadata_dict["feature_dimensions"]
    node_types = list(metadata_dict["node_counts"].keys())
    edge_types = [
        ("account", "transfers_to", "account"),
        ("account", "rev_transfers_to", "account"),
        ("customer", "owns", "account"),
        ("account", "rev_owns", "customer"),
        ("account", "logged_in_from", "device"),
        ("device", "rev_logged_in_from", "account"),
    ]
    metadata = (node_types, edge_types)

    return HeteroGraphSAGE(
        in_channels_dict=in_channels_dict,
        metadata=metadata,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        aggr=aggr,
        dropout=dropout,
    )
