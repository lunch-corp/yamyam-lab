import torch
from torch_geometric.data import Data


# Prepare the graph data
def create_graph_data(X_train, y_train, diner_mapping, user_mapping):
    diner_idx = torch.tensor([diner_mapping[x[0].item()] for x in X_train])
    reviewer_idx = torch.tensor([user_mapping[x[1].item()] for x in X_train])
    y_train = y_train.squeeze()

    # Create edges for GNN
    edge_index = torch.stack([diner_idx, reviewer_idx], dim=0)

    # Create Data object for PyTorch Geometric
    data = Data(x=torch.cat([diner_idx, reviewer_idx]), edge_index=edge_index, y=y_train)
    return data
