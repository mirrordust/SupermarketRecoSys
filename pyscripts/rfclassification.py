# encoding=utf-8
import errno
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from optparse import OptionParser

import utils


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


def feature_and_label(feature_df, label_df, subsample, label_first_date, datedelta_slots, cust_amt_thr=None,
                      cust_purch_thr=None, goods_purch_thr=None):
    # 目标顾客
    cust_label = label_df['vipno'].unique()
    cust_label = cust_label[~np.isnan(cust_label)]
    goal_cust = []
    for vipno_ in cust_label:
        subdf = feature_df[feature_df['vipno'] == vipno_]
        if cust_amt_thr is not None:
            assert cust_purch_thr is None
            if subdf.amt.sum() >= cust_amt_thr:
                goal_cust.append([vipno_])
        if cust_purch_thr is not None:
            assert cust_amt_thr is None
            n_purch = subdf.shape[0]
            n_pluno = subdf.drop_duplicates(subset=['pluno'], keep='first').shape[0]
            # n_dpt3 = subdf.drop_duplicates(subset=['dptno3'], keep='first').shape[0]
            if n_purch > 1 and n_pluno >= cust_purch_thr:
                goal_cust.append([vipno_])
    goal_cust = np.array(goal_cust)
    print 'n_goal_cust: {}'.format(goal_cust.shape[0])

    # 目标商品
    goods_label = label_df[np.in1d(label_df['vipno'], goal_cust)].pluno.unique()
    goal_goods = []
    for pluno_ in goods_label:
        subdf = feature_df[feature_df['pluno'] == pluno_]
        n_plu_purch = subdf.shape[0]
        if goods_purch_thr is not None:
            if n_plu_purch >= goods_purch_thr:
                goal_goods.append(pluno_)
        else:
            if n_plu_purch >= 4:  # default threshold
                goal_goods.append(pluno_)
    goal_goods = np.array(goal_goods)
    print 'n_goal_goods: {}'.format(goal_goods.shape[0])

    # 选取子集
    if subsample:
        if len(subsample) == 1:
            df_goal_goods = label_df[np.in1d(label_df['pluno'], goal_goods)]
            goal_cust = random_subsample(goal_cust, subsample[0])
            goal_goods = df_goal_goods[np.in1d(df_goal_goods['vipno'], goal_cust)].pluno.unique()
        else:
            goal_cust = random_subsample(goal_cust, subsample[0])
            goal_goods = random_subsample(goal_goods, subsample[1])
    print 'final # customers: {}, # goods: {}'.format(goal_cust.shape[0], goal_goods.shape[0])
    customers = label_df[np.in1d(label_df['vipno'], goal_cust)].drop_duplicates(subset='vipno', keep='first')[
        ['vipno']].reset_index(drop=True).values
    goods = label_df[np.in1d(label_df['pluno'], goal_goods)].drop_duplicates(subset='pluno', keep='first')[
        ['pluno', 'bndno', 'dptno3']].reset_index(drop=True).values
    # 构建feature
    features = utils.features(feature_df, customers, goods, label_first_date, datedelta_slots)
    print 'features finish.'
    # 构建label
    # todo: 加快速度
    labels = [((label_df['vipno'] == vip_plu_[0]) & (label_df['pluno'] == vip_plu_[1])).any() for vip_plu_ in
              features.iloc[:, 0:2].values]
    labels = np.array(labels)
    print 'labels finish.'
    return features, labels


def run(subsample=None, save_rootdir=None, cust_amt_thr=None, cust_purch_thr=None, goods_purch_thr=None):
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

    # 训练集
    print 'Training*********************************************'
    df_56 = utils.partition_by_month(df, month=[5, 6])
    df_7 = utils.partition_by_month(df, month=[7])
    training_first_date = datetime.date(2016, 7, 1)
    training_data, training_label = feature_and_label(df_56, df_7, subsample, training_first_date,
                                                      utils.datedelta_slots, cust_amt_thr=cust_amt_thr,
                                                      cust_purch_thr=cust_purch_thr, goods_purch_thr=goods_purch_thr)

    if save_rootdir is not None:
        training_data.to_csv('./{}/training_data.csv'.format(save_rootdir))
        np.savetxt('./{}/training_label.csv'.format(save_rootdir), training_label, delimiter=',')

    clf = RandomForestClassifier(n_jobs=-1, n_estimators=100).fit(training_data.values, np.array(training_label))

    # 测试集
    print 'Testing*********************************************'
    df_67 = utils.partition_by_month(df, month=[6, 7])
    df_8 = utils.partition_by_month(df, month=[8])
    testing_first_date = datetime.date(2016, 8, 1)
    testing_data, testing_label = feature_and_label(df_67, df_8, subsample, testing_first_date, utils.datedelta_slots,
                                                    cust_amt_thr=cust_amt_thr, cust_purch_thr=cust_purch_thr,
                                                    goods_purch_thr=goods_purch_thr)

    if save_rootdir is not None:
        testing_data.to_csv('./{}/testing_data.csv'.format(save_rootdir))
        np.savetxt('./{}/testing_label.csv'.format(save_rootdir), testing_label, delimiter=',')

    # 预测
    # predict_ = clf.predict(testing_data.iloc[:, 2:].values)
    # print 'predict True or False, precision', precision_score(np.array(testing_label), predict_)
    # print 'predict True or False, recall', recall_score(np.array(testing_label), predict_)
    print 'clf.classes_', clf.classes_
    # 对每个用户，预测推荐列表
    vipnos = testing_data.vipno.unique()
    results = []  # (one row: vipno, precision, recall, first_hit)
    for vipno_ in vipnos:
        idx = testing_data['vipno'] == vipno_
        idx = np.array(idx)
        sub_testing_data = testing_data[idx]
        sub_testing_label = testing_label[idx]
        predict_prob_ = clf.predict_proba(sub_testing_data.values)
        proba = predict_prob_[:, 1]
        # calculate result
        precision = utils.precision_top_k(sub_testing_label, proba)
        recall = utils.recall_top_k(sub_testing_label, proba)
        first_hit = utils.first_hit(sub_testing_label, proba)
        results.append([vipno_, precision, recall, first_hit])
    results = np.array(results)

    assert np.sum(results[:, 1] != results[:, 2]) == 0

    precisions = results[:, 1]
    report_str = 'min precision: {}, max precision: {}, mean precision: {}, median precision: {}, 70% precision: {}, ' \
                 '80% precision: {}, 90% precision: {}'.format(np.min(precisions), np.max(precisions),
                                                               np.mean(precisions), np.median(precisions),
                                                               utils.cdf_percentage(precisions, 0.7),
                                                               utils.cdf_percentage(precisions, 0.8),
                                                               utils.cdf_percentage(precisions, 0.9))
    print report_str
    if save_rootdir is not None:
        with open('./{}/report.txt'.format(save_rootdir), 'w') as f:
            f.write(report_str + '\n')
    # plot precision/recall
    X, Y = utils.cdf(precisions)
    if save_rootdir is not None:
        utils.plot_cdf(X, Y, xlabel='precision/recall', ylabel='proportion',
                       save='./{}/precision.png'.format(save_rootdir))

    # plot first hit
    first_hits = results[:, 3]
    X, Y = utils.cdf(first_hits)
    if save_rootdir is not None:
        utils.plot_cdf(X, Y, xlabel='first hit', ylabel='proportion', save='./{}/first_hit.png'.format(save_rootdir))


if __name__ == '__main__':
    import time
    import datetime

    parser = OptionParser()
    parser.add_option('-d', '--dir', dest='rootdir', help='rootdir')
    parser.add_option('-s', '--sub', dest='subsample', help='subsample')
    parser.add_option('-a', '--amt', dest='cust_amt_thr', help='cust_amt_thr')
    parser.add_option('-p', '--plu', dest='cust_purch_thr', help='cust_purch_thr')
    parser.add_option('-g', '--gpu', dest='goods_purch_thr', help='goods_purch_thr')

    # args = sys.argv
    (options, args) = parser.parse_args()

    ts = time.time()
    start = datetime.datetime.fromtimestamp(ts)
    print 'start: {}'.format(start.strftime('%Y-%m-%d %H:%M:%S'))

    # rootdir = '_'.join(args) + '_' + st
    print options
    rootdir = options.rootdir
    subsample = options.subsample
    cust_amt_thr = options.cust_amt_thr if options.cust_amt_thr is None else float(options.cust_amt_thr)
    cust_purch_thr = options.cust_purch_thr if options.cust_purch_thr is None else float(options.cust_purch_thr)
    goods_purch_thr = options.goods_purch_thr if options.goods_purch_thr is None else int(options.goods_purch_thr)

    run((float(subsample),), rootdir, cust_amt_thr=cust_amt_thr, cust_purch_thr=cust_purch_thr,
        goods_purch_thr=goods_purch_thr)
    # if len(args) == 1:
    #     print 'run with all customers & goods from label month'
    #     run(save_rootdir=rootdir)
    # else:
    #     assert len(args) <= 3
    #     if len(args) == 2:
    #         subsample = (float(args[1]),)
    #     else:
    #         subsample = (float(args[1]), float(args[2]))
    #     print 'run with subsample = {}'.format(subsample)
    #     run(subsample, save_rootdir=rootdir)

    ts = time.time()
    end = datetime.datetime.fromtimestamp(ts)
    print 'end: {}'.format(end.strftime('%Y-%m-%d %H:%M:%S'))

    print 'duration: {} seconds'.format((end - start).total_seconds())
