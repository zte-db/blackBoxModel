from tensorboardX import SummaryWriter
import torch
from model import ZScore, dataloader
from model import MLP
from dataloader import gen_train_test
import pickle
import numpy as np
import os


n_epochs = 100000

logger = SummaryWriter(log_dir="log")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


train_all_x, train_all_y, train_all_fault, test_all_x, test_all_y, test_all_fault = gen_train_test()


x_means, x_std, _ = ZScore(np.vstack((train_all_x,test_all_x)))
train_all_x = (train_all_x-x_means)/x_std
test_all_x = (test_all_x-x_means)/x_std

y_means, y_std, _ = ZScore(np.vstack((train_all_y,test_all_y)))
train_all_y = (train_all_y-y_means)/y_std
test_all_y = (test_all_y-y_means)/y_std

print(train_all_x.max(),test_all_x.max(),train_all_y.max(),test_all_y.max())


def train(model, _dataloader):
    lossFunc = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        count = 0
        for x, y in _dataloader:
            x ,y = x.to(device),y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = lossFunc(output.squeeze(), y)·
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(),3)
            optimizer.step()
            train_loss += loss.item()*x.size(0)
            count += x.size(0)
        logger.add_scalar("train loss", train_loss/count,
                          global_step=epoch)
        if epoch%1000==0:
            torch.save(model.state_dict(), "model.ckpt")
            print("train loss:", train_loss/count)
            test(model,plot=False)

        


def test(model, plot=False):
    _dataloader = dataloader(test_all_x, test_all_y, len(test_all_x))

    lossFunc = torch.nn.MSELoss()
    test_loss = 0.0
    model.eval()

    with torch.no_grad():  # 训练集中不需要反向传播
        count = 0
        for x, y in _dataloader:
            x ,y = x.to(device),y.to(device)

            outputs = model(x)
            loss = lossFunc(outputs, y)
            test_loss += loss.cpu().detach().item()*x.size(0)
            count += x.size(0)
            if plot:
                pred = outputs.squeeze().cpu().numpy().tolist()
                label = y.cpu().numpy().tolist()
                
                with SummaryWriter('log/pred') as writer:
                    for i in range(len(pred)):
                        writer.add_scalar("test pred and label",pred[i],i)
                with SummaryWriter('log/label') as writer:
                    for i in range(len(pred)):
                        writer.add_scalar("test pred and label",label[i],i)
        print("test_loss:", test_loss/ count)


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
            X[i, :], model)
        res = res + integrated_gradients
    
    
    res_importance = [round(i, 5) for i in (res/X.shape[0]).numpy().tolist()]
    sorted_id = sorted(range(len(res_importance)), key=lambda k: res_importance[k], reverse=True)
    lines = open("sql_dict.txt").readlines()
    for id in sorted_id:
        print(res_importance[id]," ",lines[id],)

    # for i,v in enumerate(res_importance):
    #     logger.add_scalar("importance",v,i)
    



if __name__ == '__main__':
    # train_all_x, train_all_y, train_all_fault, test_all_x, test_all_y, test_all_fault = gen_train_test()

    # pickle.dump([train_all_x, train_all_y, train_all_fault,
    #             test_all_x, test_all_y, test_all_fault],open("input.pkl","wb"))
    
    # model = MLP(len(train_all_x[0])).to(device)
    model = MLP(len(train_all_x[0]))

    if os.path.exists("model.ckpt"):
        model_dict = model.load_state_dict(torch.load("model.ckpt"))

    # _dataloader = dataloader(train_all_x, train_all_y,shuffle=True)
    # train(model, _dataloader)


    
    # test(model,plot=True)

    test_all_x = test_all_x.tolist()
    print(np.array(test_all_x).shape)
    X = []
    for i,data in enumerate(test_all_fault):
        if data==1:
            X.append(test_all_x[i])
    get_importance(model, X)

