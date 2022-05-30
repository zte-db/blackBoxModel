from turtle import shape
import pandas as pd
import numpy as np
import os
import random
import re
import pickle
import time
from sklearn.utils import shuffle


default_metrics_path = "/data/fault_20220417/metrics.csv"
default_sql_path = "/data/fault_20220417/log/sql_.log"
default_fault_path = "/data/fault_20220417/fault1.out"

encoded_metrics_timestamp_path = "./encode_metrics_timestamp.pkl"

def getTimeStamp(date):
    import time

    date = date.split(".")
    # dt = "2016-05-05 20:28:54.123"
    # 转换成时间数组
    timeArray = time.strptime(date[0], "%Y-%m-%d %H:%M:%S")
    # 转换成时间戳
    timestamp = time.mktime(timeArray)
    return timestamp+float(date[1])*0.001


# [[encode,cpu,timestamp]]
def read_metrics_data(fileName, sql_data, index):
    df = pd.read_csv(fileName, header=None, sep=',',
                     skiprows=6).to_numpy().tolist()

    df = [[float(item[0]), float(item[index])]
          for item in df if str(item[0]).startswith("1")]

    data = []

    j = 0
    for t in df:
        tmp = []

        while j < len(sql_data) and sql_data[j][0] <= t[0]:
            if sql_data[j][0] >= t[0]-1:    
                tmp.append(sql_data[j][1])
                
            j=j+1
        data.append([tmp,t[1],t[0]])

    print(data[0][2],data[-1][2])

    time_to_metrics = {}
    for item in data:
        time_to_metrics[item[2]]=item[1]
    

    t_sql = []
    for item in data:
        for sql in item[0]:
            t_sql.append([item[2],sql])
    

    encoded = process_sql_data(t_sql)
    

    _data = []

    for i,item in enumerate(encoded):
        _data.append([item[1],time_to_metrics[item[0]],item[0]])

    _data = sorted(_data,key=lambda x:x[2])
    return _data


# [[timestamp,sql]]
def read_sql_data(fileName):
    with open(fileName, "r") as f:
        lines = f.readlines()
        data = []
        for i,line in enumerate(lines):
            line = line.strip().split(": ")

            data.append([(getTimeStamp(line[0].split(" CST")[0])), line[-1]])

        return data


def process_sql_data(data):
    res = {}
    used_timestamp = []
    for sql in data:
        if sql[1] == "BEGIN" or sql[1] == "COMMIT" or re.match('SET (.*)', sql[1].strip()) or sql[1] == "ROLLBACK":
            continue
        if "insert" not in str(sql[1]).lower() and "update" not in str(sql[1]).lower() and "delete" not in str(sql[1]).lower() and "select" not in str(sql[1]).lower():
            continue
        
        used_timestamp.append(sql[0])
        if sql[0] not in res.keys():
            res[sql[0]] = {sql[1]: 1}
        elif sql[1] not in res[sql[0]].keys():
            res[sql[0]][sql[1]] = 1
        else:
            res[sql[0]][sql[1]] += 1

    used_timestamp = sorted(list(set(used_timestamp)))
    
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
    res = []
    for i, t in enumerate(used_timestamp):
        res.append([t, sql_encode[i]])
    
    return res


def sql_vector(data, sql_dict):
    res = np.zeros(shape=(len(data.keys()), len(sql_dict)))
    sorted_sql = sorted(list(data.items()),key=lambda item:item[0])
    for i, item in enumerate(sorted_sql):
        for sql in item[1]:
            res[i][sql_dict.index(sql)] = item[1][sql]

    return res.tolist()




def _getTimeStamp(date):
    import time

    # dt = "2016-05-05 20:28:54"
    # 转换成时间数组
    timeArray = time.strptime(date, "%Y-%m-%d %H-%M-%S")
    # 转换成时间戳
    timestamp = time.mktime(timeArray)
    return timestamp


def _get_all_data(index=6):
    # [[encode,cpu,timestamp]]
    if os.path.exists(encoded_metrics_timestamp_path):
        _data = pickle.load(open(encoded_metrics_timestamp_path, "rb"))
        
    else:
        sql_data = read_sql_data(default_sql_path)
        _data = read_metrics_data(default_metrics_path, sql_data, index)
        pickle.dump(_data, open(encoded_metrics_timestamp_path, "wb"))
    
    cur_time = _data[0][2]
    cur_index = 0
    fault = read_fault(default_fault_path)

    fault_data = []
    for one_fault in fault:
        fault_data_item = []
        while cur_time <= one_fault[3] and cur_index < len(_data)-1:
            if cur_time >= one_fault[0] and cur_time <= one_fault[1]:
                # print("sss1",_data[cur_index])
                fault_data_item.append([_data[cur_index].copy(), 0])
            elif cur_time >= one_fault[1] and cur_time <= one_fault[2]:
                # print("sss2", _data[cur_index])
                fault_data_item.append([_data[cur_index].copy(), 1])
            elif cur_time >= one_fault[2] and cur_time <= one_fault[3]:
                # print("sss3", _data[cur_index])
                fault_data_item.append([_data[cur_index].copy(), 0])

            cur_index += 1
            cur_time = _data[cur_index][2]

        fault_data.append(fault_data_item)

    return fault_data


# 
def read_fault(file):
    res = []
    with open(file, "r") as f:
        datas = f.readlines()
        for data in datas:
            t = _getTimeStamp(
                " ".join(data.strip().split("\t")[1:]).replace(":", "-"))
            res.append(t)
    _res = []
    _cur = []
    for i, item in enumerate(res):
        if i % 4 == 0 and i != 0:
            _res.append(_cur)
            _cur = [item]
        else:
            _cur.append(item)
    # print(_res)
    return _res[1:]



if __name__ == '__main__':
    sql_data = read_sql_data(default_sql_path)
    _data = read_metrics_data(default_metrics_path, sql_data, 6)
    