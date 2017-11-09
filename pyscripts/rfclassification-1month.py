# encoding=utf-8
import errno
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score

import utils


def run(subsample=None, save_rootdir=None):
    def random_subsample(array, size):
        if type(size) == float and 0 < size <= 1:
            high = int(array.shape[0] * size)
        else:
            size = int(size)
            high = size
        randidx = np.random.permutation(array.shape[0])
        high = min(high, array.shape[0])
        print 'high: {}'.format(high)
        subarray = array[randidx[: high]]
        assert subarray.shape[0] < array.shape[0]
        return subarray

    print subsample
    if save_rootdir is not None:
        try:
            os.makedirs(save_rootdir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    # df为5678月数据
    header_rows = ['uid', 'sldat', 'pno', 'cno', 'cmrid', 'vipno', 'id', 'pluno',
                   'bcd', 'pluname', 'spec', 'pkunit', 'dptno', 'dptname', 'bndno',
                   'bndname', 'qty', 'amt', 'disamt', 'ismmx', 'mtype', 'mdocno', 'isdel']
    df = pd.read_csv('./data/trade_data_20160501_20160831.csv', sep='\t', names=header_rows)
    print 'df.shape: ', df.shape
    dpt3 = df.dptno.apply(lambda no: str(no)[:3])
    df = df.assign(dptno3=dpt3.astype(int).values)

    df_678 = utils.partition_by_month(df, [6, 7, 8])
    # 训练集
    # 用户-商品对从label周提取
    df_tr_feature = utils.partition_by_date(df, start_date=datetime.date(2016, 8, 1),
                                            end_date=datetime.date(2016, 8, 17))
    print 'df_tr_feature shape: {}'.format(df_tr_feature.shape)
    df_tr_label = utils.partition_by_date(df, start_date=datetime.date(2016, 8, 18),
                                          end_date=datetime.date(2016, 8, 24))
    print 'df_tr_label shape: {}'.format(df_tr_label.shape)
    # 要预测的用户子集C=customers，商品子集G=goods
    customers = df_tr_label.drop_duplicates(subset='vipno', keep='first')[['vipno']].reset_index(drop=True).values
    goods = df_tr_label.drop_duplicates(subset='pluno', keep='first')[['pluno', 'bndno', 'dptno3']].reset_index(
        drop=True).values
    if subsample:
        print 'subsampling with {}...'.format(subsample)
        if len(subsample) == 1:
            cust_goods = df_tr_label.drop_duplicates(subset=['vipno', 'pluno'], keep='first')[
                ['vipno', 'pluno', 'bndno', 'dptno3']].reset_index(drop=True).values
            cust_goods = random_subsample(cust_goods, subsample[0])
            df_t = pd.DataFrame(data=cust_goods, columns=['vipno', 'pluno', 'bndno', 'dptno3'])
            customers = df_t.drop_duplicates(subset='vipno', keep='first')[['vipno']].reset_index(drop=True).values
            goods = df_t.drop_duplicates(subset='pluno', keep='first')[['pluno', 'bndno', 'dptno3']].reset_index(
                drop=True).values
        else:
            customers = random_subsample(customers, subsample[0])
            goods = random_subsample(goods, subsample[1])

    print '# customers: {}, # goods: {}'.format(customers.shape[0], goods.shape[0])
    training_first_date = datetime.date(2016, 8, 18)
    training_data = utils.features(df_tr_feature, customers, goods, training_first_date, utils.datedelta_slots_2_weeks,
                                   loadfromfile=False)
    if save_rootdir is not None:
        training_data.to_csv('./{}/training_data.csv'.format(save_rootdir))

    training_label = [((df_tr_label['vipno'] == vip_plu_[0]) & (df_tr_label['pluno'] == vip_plu_[1])).any() for vip_plu_
                      in training_data.iloc[:, 0:2].values]
    training_label = np.array(training_label)

    if save_rootdir is not None:
        np.savetxt('./{}/training_label.csv'.format(save_rootdir), training_label, delimiter=',')

    clf = RandomForestClassifier(n_jobs=-1, n_estimators=100).fit(training_data.iloc[:, 2:].values,
                                                                  np.array(training_label))

    # 测试集
    df_te_feature = utils.partition_by_date(df, start_date=datetime.date(2016, 8, 8),
                                            end_date=datetime.date(2016, 8, 24))
    df_te_label = utils.partition_by_date(df, start_date=datetime.date(2016, 8, 25),
                                          end_date=datetime.date(2016, 8, 31))
    testing_first_date = datetime.date(2016, 8, 25)
    customers_2 = df_te_label.drop_duplicates(subset='vipno', keep='first')[['vipno']].reset_index(drop=True).values
    goods_2 = df_te_label.drop_duplicates(subset='pluno', keep='first')[['pluno', 'bndno', 'dptno3']].reset_index(
        drop=True).values

    if subsample:
        print 'subsampling with {}...'.format(subsample)
        if len(subsample) == 1:
            cust_goods_2 = df_te_label.drop_duplicates(subset=['vipno', 'pluno'], keep='first')[
                ['vipno', 'pluno', 'bndno', 'dptno3']].reset_index(drop=True).values
            cust_goods_2 = random_subsample(cust_goods_2, subsample[0])
            df_t = pd.DataFrame(data=cust_goods_2, columns=['vipno', 'pluno', 'bndno', 'dptno3'])
            customers_2 = df_t.drop_duplicates(subset='vipno', keep='first')[['vipno']].reset_index(drop=True).values
            goods_2 = df_t.drop_duplicates(subset='pluno', keep='first')[['pluno', 'bndno', 'dptno3']].reset_index(
                drop=True).values
        else:
            customers_2 = random_subsample(customers_2, subsample[0])
            goods_2 = random_subsample(goods_2, subsample[1])

    print '# customers: {}, # goods: {}'.format(customers_2.shape[0], goods_2.shape[0])
    testing_data = utils.features(df_te_feature, customers_2, goods_2, testing_first_date,
                                  utils.datedelta_slots_2_weeks, loadfromfile=False)

    if save_rootdir is not None:
        testing_data.to_csv('./{}/testing_data.csv'.format(save_rootdir))

    testing_label = [((df_te_label['vipno'] == vip_plu_[0]) & (df_te_label['pluno'] == vip_plu_[1])).any() for vip_plu_
                     in testing_data.iloc[:, 0:2].values]
    testing_label = np.array(testing_label)

    if save_rootdir is not None:
        np.savetxt('./{}/testing_label.csv'.format(save_rootdir), testing_label, delimiter=',')

    # 预测
    predict_ = clf.predict(testing_data.iloc[:, 2:].values)
    print 'predict True or False, precision', precision_score(np.array(testing_label), predict_)
    print 'predict True or False, recall', recall_score(np.array(testing_label), predict_)

    print 'clf.classes_', clf.classes_

    predict_prob_ = clf.predict_proba(testing_data.iloc[:, 2:].values)
    proba = predict_prob_[:, 1]

    # 对每个用户，预测推荐列表
    vipnos = testing_data.vipno.unique()
    results = []  # (one row: vipno, precision, recall, first_hit)
    for vipno_ in vipnos:
        idx = testing_data['vipno'] == vipno_
        idx = np.array(idx)
        sub_testing_data = testing_data[idx]
        sub_testing_label = testing_label[idx]
        predict_prob_ = clf.predict_proba(sub_testing_data.iloc[:, 2:].values)
        proba = predict_prob_[:, 1]
        # calculate result
        precision = utils.precision_top_k(sub_testing_label, proba)
        recall = utils.recall_top_k(sub_testing_label, proba)
        first_hit = utils.first_hit(sub_testing_label, proba)
        results.append([vipno_, precision, recall, first_hit])
    results = np.array(results)

    assert np.sum(results[:, 1] != results[:, 2]) == 0

    precisions = results[:, 1]
    print 'mean precisions: {}, median precisions: {}, max precisions: {}, min precisions: {}'.format(
        np.mean(precisions), np.median(precisions), np.max(precisions), np.min(precisions))
    # precision/recall
    X, Y = utils.cdf(precisions)
    if save_rootdir is not None:
        utils.plot_cdf(X, Y, xlabel='precision/recall', ylabel='proportion',
                       save='./{}/precision.png'.format(save_rootdir))

    # first hit
    first_hits = results[:, 3]
    X, Y = utils.cdf(first_hits)
    if save_rootdir is not None:
        utils.plot_cdf(X, Y, xlabel='first hit', ylabel='proportion', save='./{}/first_hit.png'.format(save_rootdir))


if __name__ == '__main__':
    import time
    import datetime
    import cProfile

    start = time.clock()
    print 'start: {}'.format(start)
    args = sys.argv
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H:%M:%S')
    rootdir = '_'.join(args) + '_' + st
    if len(args) == 1:
        print 'run with all customers & goods from label month'
        run(save_rootdir=rootdir)
        # cProfile.run('run(save_rootdir=rootdir)', filename='./{}/cProfile.out'.format(rootdir))
    else:
        assert len(args) <= 3
        if len(args) == 2:
            subsample = (float(args[1]),)
        else:
            subsample = (float(args[1]), float(args[2]))
        print 'run with subsample = {}'.format(subsample)
        run(subsample, save_rootdir=rootdir)
        # cProfile.run('run(subsample, save_rootdir=rootdir)', filename='./{}/cProfile.out'.format(rootdir))
    end = time.clock()
    print 'end: {}, duration: {}'.format(end, end - start)
