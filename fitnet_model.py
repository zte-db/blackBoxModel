from tensorboardX import SummaryWriter
import torch
from model import dataloader
from model import MLP
from util_fcg import gen_train_test
import pickle
import numpy as np


n_epochs = 1000

logger = SummaryWriter(log_dir="log")



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
        logger.add_scalar("train loss", train_loss,
                          global_step=epoch)
        


def test(model, _dataloader):
    lossFunc = torch.nn.MSELoss()
    test_loss = 0.0
    with torch.no_grad():  # 训练集中不需要反向传播
        for x, y in _dataloader:
            outputs = model(x)
            loss = lossFunc(outputs, y)
            test_loss += loss.item()*x.size(0)
        
            pred = torch.unsqueeze(outputs, dim=1).numpy().tolist()
            label = y.numpy().tolist()
            with SummaryWriter('log/pred') as writer:
                for i in range(len(pred)):
                   writer.add_scalar("test pred and label",pred[i],i)
            with SummaryWriter('log/label') as writer:
                for i in range(len(pred)):
                    writer.add_scalar("test pred and label",label[i],i)
        print("test_loss:", test_loss)


def compute_integrated_gradient(batch_x, model):
    batch_blank = torch.zeros_like(batch_x)

    mean_grad = 0
    n = 100

    for i in range(1, n + 1):
        x = batch_blank + i / n * (batch_x - batch_blank)
        x.requires_grad = True
        y = model(x)
        (grad,) = torch.autograd.grad(y, x)
        mean_grad += grad / n

    integrated_gradients = (batch_x - batch_blank) * mean_grad

    return integrated_gradients, mean_grad


def get_importance(model, x):
    X = torch.from_numpy(np.array(x, dtype=float)).to(torch.float32)
    res = torch.zeros_like(X[0, :])

    for i in range(X.shape[0]):
        integrated_gradients, mean_grad = compute_integrated_gradient(
            X[0, :], model)
        res = res + integrated_gradients
    
    
    res_importance = [round(i, 5) for i in (res/X.shape[0]).numpy().tolist()]
    for i,v in enumerate(res_importance):
        logger.add_scalar("importance",v,i)
    



if __name__ == '__main__':
    train_all_x, train_all_y, train_all_fault, test_all_x, test_all_y, test_all_fault = gen_train_test()

    model = MLP(len(train_all_x[0]))
    _dataloader = dataloader(train_all_x, train_all_y)
    train(model, _dataloader)

    _dataloader = dataloader(test_all_x, test_all_y,len(test_all_x))
    test(model, _dataloader)

    X = []
    for i,data in enumerate(test_all_fault):
        if data==1:
            X.append(test_all_x[i])
    get_importance(model, X)

