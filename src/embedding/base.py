from abc import ABC, abstractmethod
import torch


class BaseEmbedding(ABC):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def initialize(self, **kwargs):
        raise NotImplementedError

    def train(self, model, optimizer, epoch):
        model.train()
        loader = model.loader(batch_size=128, shuffle=True)
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss /= len(loader)
        print(f"Epoch: {epoch:03d}, Loss: {total_loss:.4f}")
        return model
