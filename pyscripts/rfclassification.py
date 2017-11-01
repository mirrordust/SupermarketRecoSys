# encoding=utf-8
import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score

import utils


def run(subsample=False):
    # df为5678月数据
    header_rows = ['uid', 'sldat', 'pno', 'cno', 'cmrid', 'vipno', 'id', 'pluno',
                   'bcd', 'pluname', 'spec', 'pkunit', 'dptno', 'dptname', 'bndno',
                   'bndname', 'qty', 'amt', 'disamt', 'ismmx', 'mtype', 'mdocno', 'isdel']
    df = pd.read_csv('./data/trade_data_20160501_20160831.csv', sep='\t', names=header_rows)
    print 'df.shape: ', df.shape
    dpt3 = df.dptno.apply(lambda no: str(no)[:3])
    df = df.assign(dptno3=dpt3.astype(int).values)
    # 训练
    # 用户-商品对从label月提取
    df_56 = utils.partition_by_month(df, month=[5, 6])
    df_7 = utils.partition_by_month(df, month=[7])
    df7_custgoods = df_7.drop_duplicates(subset=['vipno', 'pluno'], keep='first')[['vipno', 'pluno']].reset_index(
        drop=True)
    if subsample:
        idx = np.random.randint(0, df7_custgoods.shape[0], int(df7_custgoods.shape[0] * 0.2))
    training_first_date = datetime.date(2016, 7, 1)
    training_data = utils.extract_features(df_56, training_first_date, utils.datedelta_slots, loadfromfile=True)
    training_label = [((df7_custgoods['vipno'] == vip_plu_[0]) & (df7_custgoods['pluno'] == vip_plu_[1])).any() for
                      vip_plu_
                      in training_data.iloc[:, 0:2].values]
    clf = RandomForestClassifier(n_jobs=-1, n_estimators=100).fit(training_data.iloc[:, 2:].values,
                                                                  np.array(training_label))
    # 测试
    df_67 = utils.partition_by_month(df, month=[6, 7])
    df_8 = utils.partition_by_month(df, month=[8])
    df8_custgoods = df_8.drop_duplicates(subset=['vipno', 'pluno'], keep='first')[['vipno', 'pluno']].reset_index(
        drop=True)
    testing_first_date = datetime.date(2016, 8, 1)
    testing_data = utils.extract_features(df_67, testing_first_date, utils.datedelta_slots, loadfromfile=True)
    testing_label = [((df8_custgoods['vipno'] == vip_plu_[0]) & (df8_custgoods['pluno'] == vip_plu_[1])).any() for
                     vip_plu_
                     in testing_data.iloc[:, 0:2].values]

    predict_ = clf.predict(testing_data.iloc[:, 2:].values)
    print 'predict True or False, precision', precision_score(np.array(testing_label), predict_)
    print 'predict True or False, recall', recall_score(np.array(testing_label), predict_)
    print 'clf.classes_', clf.classes_

    predict_prob_ = clf.predict_proba(testing_data.iloc[:, 2:].values)
    proba = predict_prob_[:, 1]
    utils.get_report(proba, np.array(testing_label))


if __name__ == '__main__':
    run()
