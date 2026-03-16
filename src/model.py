import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv,Linear

class GraphModel(torch.nn.Module):
    def __init__(self,hidden_channels: int,out_channels: int,metadata: tuple):
        """
        hidden_channels: Size of the hidden layers (e.g., 64 or 128)
        out_channels: Number of output classes (2 for Legit vs Fraud)
        metadata: Tuple of (node_types, edge_types) from our data engine
        """
        super().__init__()

        # LAYER 1: The "1-Hop" Neighborhood
        # HeteroConv allows us to apply a different convolutional layer 
        # to every different type of edge in our graph.
        
        # We use SAGEConv. The (-1, -1) is for "Lazy Initialization". 
        # PyTorch will figure out the input dimensions automatically during the first pass
        self.conv1 = HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_channels)
            for edge_type in metadata[1]
        })

        # LAYER 2: The "2-Hop" Neighborhood (Friends of Friends)
        self.conv2 = HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_channels)
            for edge_type in metadata[1]
        })

        # PREDICTION HEAD
        # A standard Linear layer to boil the 64-dimensional embedding down to 2 classes
        self.classifier = Linear(hidden_channels, out_channels)

    def forward(self,x_dict,edge_index_dict):
        """
        Forward Pass.
        x_dict: Dictionary of features for each node type.
        edge_index_dict: Dictionary of connections for each edge type.
        """

        #1. First Message passing layer
        x_dict=self.conv1(x_dict, edge_index_dict)

        # Apply activation function and dropout for regularization
        # We use a dictionary comprehension because we have to apply this to all node types
        for node_type,x in x_dict.items():
            x=F.relu(x)
            x=F.dropout(x,p=0.3,training=self.training)
            x_dict[node_type]=x

        #2. Second Message passing layer
        x_dict=self.conv2(x_dict,edge_index_dict)

        for node_type,x in x_dict.items():
            x=F.relu(x)
            x_dict[node_type]=x
        
        #3. Classification
        # We ONLY care about classifying the transactions. 
        # The 'card' nodes were just used to pass messages.
        transaction_embeddings= x_dict['transaction']
        out=self.classifier(transaction_embeddings)

        return out
    