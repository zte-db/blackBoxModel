import torch
from model import dataloader
from model import MLP
from util import get_all_data
import pickle


n_epochs = 200


def train(model, _dataloader):
    lossFunc = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    for epoch in range(n_epochs):
        train_loss = 0.0
        for x, y in _dataloader:
            optimizer.zero_grad()
            output = model(x)
            loss = lossFunc(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*x.size(0)
        print("loss:", train_loss)


def test(model, _dataloader):
    lossFunc = torch.nn.MSELoss()
    test_loss = 0.0
    with torch.no_grad():  # 训练集中不需要反向传播
        for x, y in _dataloader:
            outputs = model(x)
            print(torch.cat((outputs, torch.unsqueeze(y, dim=1)), dim=1))
            loss = lossFunc(outputs, y)
            test_loss += loss.item()*x.size(0)
        print("test_loss:", test_loss)


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = pickle.load(
        open("../dataset.pkl", "rb"))
    model = MLP(len(x_train[0]))
    _dataloader = dataloader(x_train, y_train)
    train(model, _dataloader)
    _dataloader = dataloader(x_test, y_test)
    test(model, _dataloader)
