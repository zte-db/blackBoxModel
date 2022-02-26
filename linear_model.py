from sklearn import datasets
from util import get_all_data
from model import linear_model
import random
from sklearn.metrics import mean_squared_error
import pickle

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = get_all_data(1)

    pickle.dump([x_train, x_test, y_train, y_test],
                open("../dataset.pkl", "wb"))

    
    # dataset = pickle.load(open("../dataset.pkl", "rb"))
    # print(dataset)
    lm = linear_model(x_train, y_train)
    pred = lm.predict(x_test)

    print("average cost err:", mean_squared_error(y_test, pred))
    print(pred[:20])
    print(y_test[:20])
