from time import time
from turtle import shape
import pandas as pd
import numpy as np
from model import linear_model
import os
import random
import re


def read_metrics_data(fileName, sql_data, index):
    df = pd.read_csv(fileName, header=None, sep=',').to_numpy().tolist()
    df = [[int(item[0]), float(item[index])] for item in df]

    res = []
    timestamp = list(set([item[0] for item in sql_data]))
    for item in df:
        if item[0] in timestamp:
            res.append(item)

    return [item[1] for item in res]


def read_sql_data(fileName):
    with open(fileName, "r") as f:
        lines = f.readlines()
        data = []
        for line in lines:
            line = line.strip().split("\t\t")
            data.append([int(line[0]), line[1]])
        return data


def process_sql_data(data):
    res = {}
    for sql in data:
        if sql[1] == "BEGIN" or sql[1] == "COMMIT" or re.match(r'SET.*]', sql[1].strip()):
            continue

        if sql[0] not in res.keys():
            res[sql[0]] = {sql[1]: 1}
        elif sql[1] not in res[sql[0]].keys():
            res[sql[0]][sql[1]] = 1
        else:
            res[sql[0]][sql[1]] += 1

    # get sql dicts
    sql_dict = []
    for key in res.keys():
        sql_dict.extend(list(res[key].keys()))
    sql_dict = list(set(sql_dict))

    return sql_vector(res, sql_dict)


def sql_vector(data, sql_dict):
    res = np.zeros(shape=(len(data.keys()), len(sql_dict)))
    for i, key in enumerate(sorted(data)):
        for sql in data[key]:
            res[i][sql_dict.index(sql)] = data[key][sql]

    return res.tolist()


# index 表示指标所在的索引
def get_all_data(index):
    rate = 0.8

    all_sql_encode = []
    all_metrics = []

    for idx, file in enumerate(os.listdir("../data/")):
        print("file idx:", idx)

        tps = int(file.split(".")[0][4:])
        sql_data = read_sql_data("../data/"+file)
        metrics = read_metrics_data("metrics1.csv", sql_data, index)
        sql_encode = process_sql_data(sql_data)
        sql_encode = tps*sql_encode

        all_metrics.extend(metrics)
        all_sql_encode.extend(sql_encode)

    c = list(zip(all_sql_encode, all_metrics))
    random.shuffle(c)
    all_sql_encode, all_metrics = zip(*c)

    x_train = all_sql_encode[:int(rate*len(all_metrics))]
    x_test = all_sql_encode[int(rate*len(all_metrics)):]

    y_train = all_metrics[:int(rate*len(all_metrics))]
    y_test = all_metrics[int(rate*len(all_metrics)):]

    return x_train, x_test, y_train, y_test
