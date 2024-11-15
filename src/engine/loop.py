import torch


def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)  # 예측값 계산
        loss = criterion(out[data.edge_index[1]], data.edge_attr)  # 대상 노드의 예측값과 실제 점수를 비교
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


@torch.no_grad()
def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0

    for data in val_loader:
        out = model(data)
        loss = criterion(out[data.edge_index[1]], data.edge_attr)
        total_loss += loss.item()

    return total_loss / len(val_loader)
