# encoding=utf-8

import math
import pickle
import random
from collections import defaultdict

import numpy


def inner_product(v1, v2):
    return numpy.dot(v1, v2)


def predict_score(pu, qi):
    p_score = inner_product(pu, qi)
    return float(p_score)


# old
def SVD(config, train_file, model_save_file):
    # get the configure
    learn_rate = config.learn_rate
    regularization = config.regularization
    factor_num = config.factor_num
    qi = defaultdict(lambda: numpy.array([random.random() for i in range(factor_num)]))
    pu = defaultdict(lambda: numpy.array([random.random() for i in range(factor_num)]))
    print("initialization end\nstart training\n")
    rmse = 10000
    # train model
    for step in range(100):
        print('step==>{0}'.format(step))

        fi = open(train_file, 'r')
        for line in fi:
            arr = line.split(',')
            uid = int(arr[0].strip())
            iid = int(arr[1].strip())
            score = float(arr[2].strip())

            prediction = predict_score(pu[uid], qi[iid])

            eui = score - prediction

            for k in range(factor_num):
                temp = pu[uid][k]  # attention here, must save the value of pu before updating
                pu[uid][k] += float(learn_rate * (eui * qi[iid][k] - regularization * pu[uid][k]))
                qi[iid][k] += float(learn_rate * (eui * temp - regularization * qi[iid][k]))
        fi.close()
        learn_rate *= 0.9

        rmse_train = Validate(train_file, pu, qi)
        if rmse_train > rmse:
            break
        rmse = rmse_train
        print('rmse = {0}'.format(rmse))
    fo = open(model_save_file, 'wb')

    pickle.dump(dict(qi), fo, True)
    pickle.dump(dict(pu), fo, True)

    fo.close()
    rmse = Validate(train_file, pu, qi)
    return rmse, pu, qi


# old
def Validate(test_file, pu, qi):
    rmse = 0.0
    cnt = 0
    fi = open(test_file, 'r')
    fo = open('predictFile.txt', 'w')
    for line in fi:
        arr = line.split(',')
        uid = int(arr[0].strip())
        iid = int(arr[1].strip())
        rate = float(arr[2].strip())
        try:
            pscore = predict_score(pu[uid], qi[iid])
            rmse += (rate - pscore) * (rate - pscore)
            fo.write('{0},{1},{2},{3}\n'.format(uid, iid, rate, pscore))
            cnt += 1
        except Exception:
            pass
    return math.sqrt(rmse / cnt)


class Config:
    def __init__(self, factor_num, learn_rate, regularization):
        self.factor_num = factor_num
        self.learn_rate = learn_rate
        self.regularization = regularization


def svd_features(config, data):
    """Generate svd features.

    :param config: svd mf params
    :param data: in 2-d np.array, each row has [uid, iid, rate]
    :return: rmse, user_vector dict, item_vector dict
    """
    # get the configure
    learn_rate = config.learn_rate
    regularization = config.regularization
    factor_num = config.factor_num
    qi = defaultdict(lambda: numpy.array([random.random() for i in range(factor_num)]))
    pu = defaultdict(lambda: numpy.array([random.random() for i in range(factor_num)]))
    print("initialization end\nstart training\n")
    rmse = 10000
    # train model
    for step in range(100):
        print('step==>{0}'.format(step))
        # todo:perfect this
        for line in data:
            uid = int(line[0])
            iid = int(line[1])
            score = float(line[2])

            prediction = predict_score(pu[uid], qi[iid])

            eui = score - prediction

            for k in range(factor_num):
                temp = pu[uid][k]  # attention here, must save the value of pu before updating
                pu[uid][k] += float(learn_rate * (eui * qi[iid][k] - regularization * pu[uid][k]))
                qi[iid][k] += float(learn_rate * (eui * temp - regularization * qi[iid][k]))
        learn_rate *= 0.9

        rmse_train = validate(data, pu, qi)
        if rmse_train > rmse:
            break
        rmse = rmse_train

    rmse = validate(data, pu, qi)
    return rmse, pu, qi


def validate(data, pu, qi):
    rmse = 0.0
    cnt = 0
    for line in data:
        uid = int(line[0])
        iid = int(line[1])
        rate = float(line[2])
        try:
            pscore = predict_score(pu[uid], qi[iid])
            rmse += (rate - pscore) * (rate - pscore)
            cnt += 1
        except Exception:
            pass
    return math.sqrt(rmse / cnt)
