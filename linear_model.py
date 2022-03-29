from sklearn import datasets
from util_fcg import gen_train_test
from model import linear_model
import random
from sklearn.metrics import mean_squared_error
import pickle

if __name__ == '__main__':
    train_all_x, train_all_y, train_all_fault, test_all_x, test_all_y, test_all_fault = gen_train_test()

    pickle.dump([train_all_x, train_all_y, train_all_fault, test_all_x, test_all_y, test_all_fault],
                open("../dataset.pkl", "wb"))

    
    # dataset = pickle.load(open("../dataset.pkl", "rb"))
    # print(dataset)
    lm = linear_model(train_all_x, train_all_y)
    pred = lm.predict(test_all_x)

    print("average cost err:", mean_squared_error(test_all_y, pred))
    print(pred[:20])
    print(test_all_y[:20])
