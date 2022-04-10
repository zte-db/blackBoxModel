from time import time
from turtle import shape
import pandas as pd
import numpy as np
from model import linear_model
import os
import random
import re
import pickle


def getTimeStamp(date):
    import time

    #dt = "2016-05-05 20:28:54"
    #转换成时间数组
    timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    #转换成时间戳
    timestamp = time.mktime(timeArray)
    return timestamp


def read_metrics_data(fileName, sql_data, index):
    df = pd.read_csv(fileName, header=None, sep=',').to_numpy().tolist()
    df = [[int(item[0]), float(item[index])] for item in df]

    res = []
    timestamp = list(set([item[0] for item in sql_data]))
    for item in df:
        if item[0] in timestamp:
            res.append(item)

    return res


def read_sql_data(fileName):
    with open(fileName, "r") as f:
        lines = f.readlines()
        data = []
        for line in lines:
            line = line.strip().split(": ")
            data.append([int(getTimeStamp(line[0].split(".")[0])), line[-1]])
        return data


def process_sql_data(data):
    res = {}
    used_timestamp = []
    for sql in data:
        if sql[1] == "BEGIN" or sql[1] == "COMMIT" or re.match('SET (.*)', sql[1].strip()) or sql[1] == "ROLLBACK":
            continue
        used_timestamp.append(sql[0])
        if sql[0] not in res.keys():
            res[sql[0]] = {sql[1]: 1}
        elif sql[1] not in res[sql[0]].keys():
            res[sql[0]][sql[1]] = 1
        else:
            res[sql[0]][sql[1]] += 1

    used_timestamp = list(set(used_timestamp))
    print(used_timestamp[:20])

    # get sql dicts
    sql_dict = []
    for key in res.keys():
        sql_dict.extend(list(res[key].keys()))
    sql_dict = list(set(sql_dict))

    with open("sql_dict.txt", "w") as f:
        for sql in sql_dict:
            f.write(sql+"\n")
        f.close()
    # combine timestamp
    sql_encode = sql_vector(res, sql_dict)
    res=[]
    for i,t in enumerate(used_timestamp):
        res.append([t,sql_encode[i]])
    return res



def sql_vector(data, sql_dict):
    res = np.zeros(shape=(len(data.keys()), len(sql_dict)))
    for i, key in enumerate(sorted(data)):
        for sql in data[key]:
            res[i][sql_dict.index(sql)] = data[key][sql]

    return res.tolist()



def _getTimeStamp(date):
    import time

    #dt = "2016-05-05 20:28:54"
    # 转换成时间数组
    timeArray = time.strptime(date, "%Y-%m-%d %H-%M-%S")
    # 转换成时间戳
    timestamp = time.mktime(timeArray)
    return timestamp



def _get_all_data(index,file):
    rate = 0.8

    # sql_data = read_sql_data(file)
    # f = open("/home/zte/lzd/data/sql_data.pkl", "wb")
    # pickle.dump(sql_data, f)
    # f.close()

    sql_data = pickle.load(open("/home/zte/lzd/data/sql_data.pkl", "rb"))

    metrics = read_metrics_data("/home/zte/lzd/data/metrics1.csv", sql_data, index)
    sql_encode = process_sql_data(sql_data)

    sql_time = list(set([item[0] for item in sql_encode]))
    all_data = []
    i = 0
    j = 0
    # assert len(sql_time)==len(sql_encode)
    while i<len(metrics) and j<len(sql_time):
        if metrics[i][0]==sql_time[j]:
            all_data.append([metrics[i][0],metrics[i][1],sql_encode[j]])
            i+=1
            j+=1
        elif metrics[i][0] > sql_time[j]:
            j+=1
        else:
            i+=1
    
    print("all data number:",len(all_data))

    cur_time  = all_data[0][0]
    cur_index = 0
    fault = read_fault("/home/zte/lzd/data/fault3.res")

    fault_data = []
    for one_fault in fault:
        fault_data_item = []
        while cur_time<=one_fault[3] and cur_index < len(all_data)-1:
            if cur_time>=one_fault[0] and cur_time<=one_fault[1]:
                fault_data_item.append([all_data[cur_index],0])
            elif cur_time >= one_fault[1] and cur_time <= one_fault[2]:
                fault_data_item.append([all_data[cur_index], 1])
            elif cur_time >= one_fault[2] and cur_time <= one_fault[3]:
                fault_data_item.append([all_data[cur_index], 0])

            cur_index+=1
            cur_time=all_data[cur_index][0]

        fault_data.append(fault_data_item)

    return fault_data

def read_fault(file):
    res = []
    with open(file,"r") as f:
        datas = f.readlines()
        for data in datas:
            t = _getTimeStamp(" ".join(data.strip().split("\t")[1:]).replace(":","-"))
            res.append(t)
    _res = []
    _cur = []
    for i,item in enumerate(res):
        if i%4==0 and i!=0:
            _res.append(_cur)
            _cur=[item]
        else:
            _cur.append(item)
    return _res[1:]


def gen_train_test():
    rate = 0.8
    all_data = _get_all_data(6, "/home/zte/lzd/data/sql.log")
    import random
    random.shuffle(all_data)
    
    train = all_data[:int(len(all_data)*rate)]
    test = all_data[int(len(all_data)*rate):]

    train_all_x,train_all_y,train_all_fault = _get(train)
    test_all_x,test_all_y,test_all_fault = _get(test)
    return train_all_x, train_all_y, train_all_fault, test_all_x, test_all_y, test_all_fault
    

def _get(all_data):
    all_x = []
    all_y = []
    all_fault = []
    for item in all_data:
        for item1 in item:
            all_x.append(item1[0][2][1])
            all_y.append(item1[0][1])
            all_fault.append(item1[1])
    return all_x,all_y,all_fault

if __name__=='__main__':
    #print(read_fault("fault3.res"))
    # train_all_x, train_all_y, train_all_fault, test_all_x, test_all_y, test_all_fault = gen_train_test()
    # print(train_all_x[0], train_all_y[0], train_all_fault[0],
    #       test_all_x[0], test_all_y[0], test_all_fault[0])
    fault = read_fault("/home/zte/lzd/data/fault3.res")
    obj = {"heavy workload":[]}
    f = open("fault.pkl","wb")
    for item in fault:
        obj["heavy workload"].append(item[1:3])
    import pickle
    pickle.dump(obj,f)
    f.close()



