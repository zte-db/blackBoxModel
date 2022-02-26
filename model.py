from turtle import forward
import numpy as np
from sklearn.linear_model import LinearRegression
import torch
import torch.utils.data as Data


def linear_model(X, Y):
    reg = LinearRegression().fit(X, Y)
    return reg


class MLP(torch.nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 16)
        self.fc2 = torch.nn.Linear(16, 8)
        self.fc3 = torch.nn.Linear(8, 1)

    def forward(self, input):
        # input [samples,input_dim]
        m = torch.nn.ReLU()
        out = m(self.fc1(input))
        out = m(self.fc2(out))
        out = m(self.fc3(out))
        return out


def dataloader(X, Y):
    X = torch.from_numpy(np.array(X, dtype=float)).to(torch.float32)
    Y = torch.from_numpy(np.array(Y, dtype=float)).to(torch.float32)

    dataset = Data.TensorDataset(X, Y)
    return Data.DataLoader(
        dataset=dataset,
        batch_size=8,
        shuffle=True,
    )
