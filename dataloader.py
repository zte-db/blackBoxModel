from turtle import shape
import pandas as pd
import numpy as np
import os
import random
import re
import pickle
import time
from util import _get_all_data




def gen_train_test():

    rate = 0.8
    all_data = _get_all_data(75)

    # print(type(all_data),len(all_data))
    # print(all_data[0])
    random.seed(time.time())
    random.shuffle(all_data)

    train = all_data[:int(len(all_data)*rate)]
    test = all_data[int(len(all_data)*rate):]

    train_all_x, train_all_y, train_all_fault = _get(train)
    test_all_x, test_all_y, test_all_fault = _get(test)
    return np.array(train_all_x), np.array(train_all_y).reshape((-1, 1)), train_all_fault, np.array(test_all_x), np.array(test_all_y).reshape((-1, 1)), test_all_fault


def _get(all_data):
    all_x = []
    all_y = []
    all_fault = []
    for item in all_data:
        for item1 in item:
            all_x.append(item1[0][0])
            all_y.append(item1[0][1])
            all_fault.append(item1[1])
    return all_x, all_y, all_fault

if __name__ == '__main__':
    # print(read_fault("fault3.res"))
    # train_all_x, train_all_y, train_all_fault, test_all_x, test_all_y, test_all_fault = gen_train_test()
    # print(train_all_x[0], train_all_y[0], train_all_fault[0],
    #       test_all_x[0], test_all_y[0], test_all_fault[0])
    # fault = read_fault("../data/fault3.res")
    # obj = {"heavy workload":[]}
    # f = open("fault.pkl","wb")
    # for item in fault:
    #     obj["heavy workload"].append(item[1:3])
    # import pickle
    # pickle.dump(obj,f)
    # f.close()
    # print(getTimeStamp("2016-05-05 20:28:54"))
    # path = "../data/"
    # data = read_sql_data(path+"sql.log")

    # # print(data)
    # _data = read_metrics_data(path+"metrics.csv",data,6)
    # print(_data[:2])

    # pickle.dump(_data,open("encode_metrics_timestamp.pkl","wb"))

    # print(data[:2])
    # print(data[:5])
    # fault = read_fault(path+"fault.out")
    # print(fault[:2])
    # train_all_x, train_all_y, train_all_fault, test_all_x, test_all_y, test_all_fault = gen_train_test()
    # print(train_all_x.shape)
    pass


# s = """
# "Time","usr","sys","idl","wai","stl","read","writ","recv","send","in","out","used","free","buff","cach","int","csw","run","blk","new","1m","5m","15m","used","free","153","154","156","read","writ","int","csw","#aio","files","inodes","msg","sem","shm","pos","lck","rea","wri","raw","tot","tcp","udp","raw","frg","lis","act","syn","tim","clo","lis","act","dgm","str","lis","act","majpf","minpf","alloc","free","steal","scanK","scanD","pgoru","astll","d32F","d32H","normF","normH","size","grow","insert","update","delete","Conn","%Con","Act","LongQ","LongX","Idl","LIdl","LWait","SQLs1","SQLs3","SQLs5","Xact1","Xact3","Locks","comm","roll","clean","back","alloc","heapr","heaph","ratio","shared_buffers","work_mem","bgwriter_delay","max_connections","autovacuum_work_mem","temp_buffers","autovacuum_max_workers","maintenance_work_mem","checkpoint_timeout","max_wal_size","checkpoint_completion_target","wal_keep_segments","wal_segment_size"
# """
# l = list(s.strip().split(","))
# print(l)
# print(l.index('"insert"'))