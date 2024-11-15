from __future__ import annotations

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch_geometric.data import DataLoader

from data import load_and_prepare_graph_data
from engine.loop import train, validate
from model.gnn import GNNModel

# from preprocess.preprocess import train_test_split_stratify


@hydra.main(config_path="../config/", config_name="train", version_base="1.2.0")
def _main(cfg: DictConfig):
    # load dataset
    data = load_and_prepare_graph_data(cfg.data.test_size, cfg.data.min_reviews)

    # 모델 초기화
    model = GNNModel(num_node_features=data.x.shape[1], hidden_channels=16).to(cfg.model.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()  # 예측 점수와 실제 점수 간의 MSE를 손실 함수로 사용
    train_loader = DataLoader(data, batch_size=32, shuffle=True)
    val_loader = DataLoader(data, batch_size=32, shuffle=False)

    # 학습 루프
    best_val_loss = float("inf")

    for epoch in range(1, cfg.models.epochs + 1):
        train_loss = train(model, train_loader, optimizer, criterion)
        val_loss = validate(model, val_loader, criterion)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")

        print(f"Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")


if __name__ == "__main__":
    _main()
