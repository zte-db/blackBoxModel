from turtle import forward
import numpy as np
from sklearn.linear_model import LinearRegression
import torch
import torch.utils.data as Data


def linear_model(X, Y):
    reg = LinearRegression().fit(X, Y)
    print("coefficients:", reg.coef_.tolist())
    return reg


class MLP(torch.nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 64)
        self.fc4 = torch.nn.Linear(64,1)
        self.dropout = torch.nn.Dropout(0.5)
    def forward(self, input):
        # input [samples,input_dim]
        m = torch.nn.ReLU()
        out = m(self.fc1(input))
        out = self.dropout(m(self.fc2(out)))
        out = self.dropout(m(self.fc3(out)))
        out = self.fc4(out)
        return out


def dataloader(X, Y, batch_size = 1024,shuffle=True):
    X = torch.from_numpy(X).to(torch.float32)
    Y = torch.from_numpy(Y).to(torch.float32).squeeze()

    dataset = Data.TensorDataset(X, Y)
    return Data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

# x,y np.array
def ZScore(x):
    import numpy as np

    means = np.mean(x,axis=0)
    std = np.std(x,axis=0)

    return means,std,(x-means)/std


if __name__ == '__main__':
    x = [[1,2,3],[1,2,3],[2,4,6]]
    means,std,x = ZScore(x)
    
