import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv, Linear

class GraphModel(torch.nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int, metadata: tuple,
                 num_layers: int = 2, dropout: float = 0.3,
                 residual: bool = True, norm: bool = True,
                 target_node_type: str = "account", aggr: str = "mean"):
        """
        hidden_channels: Size of the hidden layers (e.g., 64 or 128)
        out_channels: Number of output classes (2 for normal vs anomalous)
        metadata: Tuple of (node_types, edge_types) from our data engine
        num_layers: Number of message-passing hops
        residual / norm: skip connections and per-node-type LayerNorm around every conv
        target_node_type: the node type that is classified (any others only pass messages)
        aggr: how SAGEConv aggregates neighbor messages (mean, sum, max)
        """
        super().__init__()
        node_types, edge_types = metadata
        self.dropout = dropout
        self.residual = residual
        self.target_node_type = target_node_type

        # Every node type has its own feature size, so first project each into a shared
        # hidden space. (-1) lets PyG infer the input size on the first forward pass.
        self.proj = torch.nn.ModuleDict({t: Linear(-1, hidden_channels) for t in node_types})

        # The "k-hop" neighborhood. HeteroConv applies a different SAGEConv
        # to every edge type and sums the results per destination node type.
        self.convs = torch.nn.ModuleList([
            HeteroConv({et: SAGEConv((-1, -1), hidden_channels, aggr=aggr) for et in edge_types})
            for _ in range(num_layers)
        ])
        self.norms = torch.nn.ModuleList([
            torch.nn.ModuleDict({t: torch.nn.LayerNorm(hidden_channels) for t in node_types})
            if norm else torch.nn.ModuleDict()
            for _ in range(num_layers)
        ])

        # PREDICTION HEAD on the target-node embeddings
        self.classifier = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        """
        x_dict: Dictionary of features for each node type.
        edge_index_dict: Dictionary of connections for each edge type
                         (edge_index tensors, or transposed SparseTensor adjacencies).
        """
        x_dict = {t: F.relu(self.proj[t](x)) for t, x in x_dict.items()}

        for conv, norms in zip(self.convs, self.norms):
            out_dict = conv(x_dict, edge_index_dict)
            new_dict = dict(x_dict)
            for t, h in out_dict.items():
                if t in norms:
                    h = norms[t](h)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                if self.residual:
                    h = h + x_dict[t]
                new_dict[t] = h
            x_dict = new_dict

        # We ONLY classify the target nodes; any other node types just pass messages.
        return self.classifier(x_dict[self.target_node_type])


class EnsembleModel(torch.nn.Module):
    """Averages the fraud probabilities of several GraphModels (a seed ensemble).
    Returns log-probabilities so that softmax(output) is the averaged probability,
    which keeps it a drop-in replacement for a single GraphModel."""
    def __init__(self, members):
        super().__init__()
        self.members = torch.nn.ModuleList(members)

    def forward(self, x_dict, edge_index_dict):
        probs = torch.stack([F.softmax(m(x_dict, edge_index_dict), dim=1) for m in self.members])
        return probs.mean(dim=0).clamp_min(1e-9).log()
