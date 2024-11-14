import torch
from torch_geometric.nn import GCNConv


class ReviewGNN(torch.nn.Module):
    def __init__(self, num_diners, num_reviewers, hidden_dim=32):
        super(ReviewGNN, self).__init__()
        # Embedding layers for diners and reviewers
        self.diner_embedding = torch.nn.Embedding(num_diners, hidden_dim)
        self.reviewer_embedding = torch.nn.Embedding(num_reviewers, hidden_dim)

        # GCN or GAT layers
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Prediction layer
        self.fc = torch.nn.Linear(hidden_dim * 2, 1)

    def forward(self, diner_idx, reviewer_idx, edge_index):
        # Get embeddings for diners and reviewers
        diner_emb = self.diner_embedding(diner_idx)
        reviewer_emb = self.reviewer_embedding(reviewer_idx)

        # Apply GNN layers
        x = torch.cat([diner_emb, reviewer_emb], dim=0)
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()

        # Concatenate diner and reviewer embeddings for prediction
        diner_reviewer_emb = torch.cat([diner_emb, reviewer_emb], dim=1)
        return self.fc(diner_reviewer_emb).squeeze()
