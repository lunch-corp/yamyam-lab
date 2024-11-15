import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GNNModel(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)  # 최종 출력 차원을 1로 설정하여 예측 점수 산출

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
