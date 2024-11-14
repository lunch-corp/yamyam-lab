from __future__ import annotations

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from data.dataset import create_graph_data
from model.gnn import ReviewGNN
from preprocess.preprocess import train_test_split_stratify


@hydra.main(config_path="../config/", config_name="train", version_base="1.2.0")
def _main(cfg: DictConfig):
    # load dataset
    data = train_test_split_stratify(
        test_size=cfg.data.test_ratio,
        min_reviews=cfg.data.min_reviews,
        X_columns=cfg.data.X_columns,
        y_columns=cfg.data.y_columns,
    )

    graph_data = create_graph_data(data["X_train"], data["y_train"], data["diner_mapping"], data["user_mapping"])
    model = ReviewGNN(num_diners=data["num_diners"], num_reviewers=data["num_users"])

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.models.lr)
    model.train()
    for epoch in range(cfg.models.epochs):
        optimizer.zero_grad()
        out = model(graph_data.x, graph_data.edge_index)
        loss = F.mse_loss(out, graph_data.y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")


if __name__ == "__main__":
    _main()
