# encoding=utf-8
import datetime
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
import sys


def test_DISPLAY():
    return os.getenv("DISPLAY")


def backward_date_slots(base_datetime, datedelta_slots):
    date_slots = []
    for slot in datedelta_slots:
        dates = [base_datetime - datetime.timedelta(days=i) for i in slot]
        dates.reverse()
        date_slots.append(tuple(dates))
    return date_slots


def partition_by_month(df, month):
    def get_month(v):
        dt = datetime.datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
        return dt.month

    month = np.array(month)
    df_months = map(get_month, df.sldat.values)
    #     mask = np.array(df_months) == month
    mask = np.in1d(np.array(df_months), month)
    cdf = df.copy()
    return cdf[mask]


def partition_by_date(df, start_date, end_date):
    def in_range(date, s_date=start_date, e_date=end_date):
        return s_date <= date <= e_date

    df_dates = map(get_date, df.sldat.values)
    mask = map(in_range, df_dates)
    mask = np.array(mask)
    # mask_1 = map(lambda dt: dt >= start_date, df_dates)
    # mask_2 = map(lambda dt: dt <= end_date, df_dates)
    # mask = np.array(mask_1) & np.array(mask_2)
    cdf = df.copy()
    return cdf[mask]


def label_encode(column, return_dict=False):
    le = preprocessing.LabelEncoder()
    le.fit(column)
    trans_dict = {}
    for idx, v in enumerate(le.classes_):
        trans_dict[idx] = v
    if return_dict:
        return le.transform(column), trans_dict
    else:
        return le.transform(column)


def get_datetime(v):
    dt = datetime.datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
    return dt


def get_date(v):
    dt = datetime.datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
    return dt.date()


# 距离预测日期前 第1天、第2天、第3天、前3天、前5天、前7天、前10天、前14天、前21天、前28天、前40天、前50天、前59天 被购买次数
datedelta_slots = np.array(
    [(1, 1), (2, 2), (3, 3), (1, 3), (1, 5), (1, 7), (1, 10), (1, 14), (1, 21), (1, 28), (1, 40), (1, 50), (1, 59)])

datedelta_slots_2_weeks = np.array([(1, 1), (2, 2), (3, 3), (1, 3), (1, 5), (1, 7), (1, 10), (1, 14)])


def customer_features(source_df, customers, predict_first_date, datedelta_slots):
    cust_behavs = []
    date_slots = backward_date_slots(predict_first_date, datedelta_slots)
    last_date_slot = backward_date_slots(predict_first_date, [np.arange(datedelta_slots[-1][1]) + 1])[0]
    print '# customers:', customers.shape[0]
    for cust_ in customers:
        vipno_ = cust_[0]
        df_cust = source_df[source_df['vipno'] == vipno_]
        if df_cust.shape[0] == 0:
            # todo：无购买的最近一次购买记录间隔天数怎么确定
            behav = np.concatenate(([vipno_], np.zeros(len(datedelta_slots) + len(last_date_slot)), [999]))
        else:
            # 不同时间段内购买商品数
            ds = map(get_date, df_cust.sldat)
            ds = np.array(ds)
            # feature vector 1
            # purch_count = [np.sum(np.in1d(ds, dates)) for dates in date_slots]
            purch_count = [np.sum(np.logical_and(ds >= dates[0], ds <= dates[1])) for dates in date_slots]
            # 前59天有购买行为的日期
            day_purch_or_not = np.in1d(last_date_slot, ds)
            day_purch_or_not = day_purch_or_not.astype(int)  # feature verctor 2
            # 最后一次购买的日期距预测日期的天数
            last_purch_day = max(ds)
            min_daydelta = abs((last_purch_day - predict_first_date).days)  # feature vector 3
            # 合并(第一列为vipno)
            behav = np.concatenate(([vipno_], purch_count, day_purch_or_not, [min_daydelta]))
        cust_behavs.append(behav)
    return np.array(cust_behavs)


def goods_features(source_df, goods, predict_first_date, datedelta_slots):
    goods_behavs = []
    date_slots = backward_date_slots(predict_first_date, datedelta_slots)
    print '# goods:', goods.shape[0]
    i = 0
    # goods = ['pluno', 'bndno', 'dptno3']
    for goods_ in goods:
        if i == int(0.25 * goods.shape[0]):
            print '25% finish'
        elif i == int(0.5 * goods.shape[0]):
            print '50% finish'
        elif i == int(0.75 * goods.shape[0]):
            print '75% finish'
        i = i + 1
        # 商品被购买次数
        pluno_ = goods_[0]
        df_goods = source_df[source_df['pluno'] == pluno_]
        if df_goods.shape[0] == 0:
            purch_goods_count = np.zeros(len(datedelta_slots))
        else:
            ds1 = map(get_date, df_goods.sldat)
            ds1 = np.array(ds1)
            # feature vector 1
            purch_goods_count = [np.sum(np.logical_and(ds1 >= dates[0], ds1 <= dates[1])) for dates in date_slots]
        # todo:加快速度：先计算unique的dptnp3再merge
        # 商品种类被购买次数
        dptno3 = goods_[2]
        df_cates = source_df[source_df['dptno3'] == dptno3]
        if df_cates.shape[0] == 0:
            purch_cates_count = np.zeros(len(datedelta_slots))
        else:
            ds2 = map(get_date, df_cates.sldat)
            ds2 = np.array(ds2)
            # feature vector 2
            purch_cates_count = [np.sum(np.logical_and(ds2 >= dates[0], ds2 <= dates[1])) for dates in date_slots]
        # 合并(第一列为pluno)
        behav = np.concatenate((goods_, purch_goods_count, purch_cates_count))
        goods_behavs.append(behav)
    return np.array(goods_behavs)


def custs_goods_features(source_df, customers, goods, predict_first_date, datedelta_slots):
    custgoods_behavs = []
    # df_custgoods_pairs = df.drop_duplicates(subset=['vipno', 'pluno'], keep='first')[
    #     ['vipno', 'pluno', 'dptno3']]
    date_slots = backward_date_slots(predict_first_date, datedelta_slots)
    n_loop = customers.shape[0] * goods.shape[0]
    print '# custgoods_pairs: ', n_loop
    i = 0
    for cust_ in customers:
        vipno_ = cust_[0]
        df_cust = source_df[source_df['vipno'] == vipno_]
        # goods = ['pluno', 'bndno', 'dptno3']
        for goods_ in goods:
            if i == int(0.25 * n_loop):
                print '25% finish'
            elif i == int(0.5 * n_loop):
                print '50% finish'
            elif i == int(0.75 * n_loop):
                print '75% finish'
            i = i + 1
            pluno_ = goods_[0]
            dptno3_ = goods_[2]
            # 商品被用户购买次数
            df_cust_goods = df_cust[df_cust['pluno'] == pluno_]
            if df_cust_goods.shape[0] == 0:
                purch_goods_count = np.zeros(len(datedelta_slots))
            else:
                ds1 = map(get_date, df_cust_goods.sldat)
                ds1 = np.array(ds1)
                # feature vector 1
                purch_goods_count = [np.sum(np.logical_and(ds1 >= dates[0], ds1 <= dates[1])) for dates in date_slots]
            # 商品种类被用户购买次数
            df_cust_cates = df_cust[df_cust['dptno3'] == dptno3_]
            if df_cust_cates.shape[0] == 0:
                purch_cates_count = np.zeros(len(datedelta_slots))
            else:
                ds2 = map(get_date, df_cust_cates.sldat)
                ds2 = np.array(ds2)
                # feature vector 2
                purch_cates_count = [np.sum(np.logical_and(ds2 >= dates[0], ds2 <= dates[1])) for dates in date_slots]
            # 合并
            behav = np.concatenate(([vipno_, pluno_], purch_goods_count, purch_cates_count))
            custgoods_behavs.append(behav)
    return np.array(custgoods_behavs)


def features(source_df, customers, goods, predict_first_date, datedelta_slots):
    """特征提取

    用户：
    1. 距离预测日期前 第1天、第2天、第3天、前3天、前5天、前7天、前10天、前14天、前21天、前28天、前40天、前50天、前59天 购买次数
    2. 前59天有购买行为的日期
    3. 最后一次购买的日期距预测日期的天数

    商品：
    1. 距离预测日期前 第1天、第2天、第3天、前3天、前5天、前7天、前10天、前14天、前21天、前28天、前40天、前50天、前59天 被购买次数
    2. 距离预测日期前 第1天、第2天、第3天、前3天、前5天、前7天、前10天、前14天、前21天、前28天、前40天、前50天、前59天 该商品种类 被购买次数

    用户-商品：
    1. 距离预测日期前 第1天、第2天、第3天、前3天、前5天、前7天、前10天、前14天、前21天、前28天、前40天、前50天、前59天
    商品G被用户U 购买次数
    2. 距离预测日期前 第1天、第2天、第3天、前3天、前5天、前7天、前10天、前14天、前21天、前28天、前40天、前50天、前59天
    商品种类C被用户U 购买次数
    """
    timestr = '{}_{}_{}'.format(predict_first_date.year, predict_first_date.month, predict_first_date.day)
    # customers
    cust_f = customer_features(source_df, customers, predict_first_date, datedelta_slots)
    print 'cust features finish'
    # pd.DataFrame(data=cust_f).to_csv('cust_f_{}.csv'.format(timestr), index=False)
    # goods
    goods_f = goods_features(source_df, goods, predict_first_date, datedelta_slots)
    print 'goods features finish'
    # pd.DataFrame(data=goods_f).to_csv('goods_f_{}.csv'.format(timestr), index=False)
    # custs-goods
    custgoods_f = custs_goods_features(source_df, customers, goods, predict_first_date, datedelta_slots)
    print 'custgoods features finish'
    # pd.DataFrame(data=custgoods_f).to_csv('custgoods_f_{}.csv'.format(timestr), index=False)

    cust_df = pd.DataFrame(data=cust_f)
    cust_df.columns = ['vipno'] + ['cust_' + str(i) for i in range(1, cust_df.shape[1])]
    goods_df = pd.DataFrame(data=goods_f)
    goods_df.columns = ['pluno'] + ['goods_' + str(i) for i in range(1, goods_df.shape[1])]
    custgoods_df = pd.DataFrame(data=custgoods_f)
    custgoods_df.columns = ['vipno', 'pluno'] + ['custgoods_' + str(i) for i in range(2, custgoods_df.shape[1])]

    training_data = custgoods_df.iloc[:, 0:2]
    merge1 = pd.merge(training_data, cust_df, sort=False, on='vipno', how='left', copy=False)
    merge2 = pd.merge(merge1, goods_df, sort=False, on='pluno', how='left', copy=False)
    merge2 = merge2.fillna(-1)  # todo:改进：填充nan的bndno
    merge3 = pd.merge(merge2, custgoods_df, sort=False, on=['vipno', 'pluno'], how='left', copy=False)
    return merge3


# Report part

def precision_at_k(true_label, predict_prob, k):
    """
    Precision@k = (# of recommended items @k that are relevant) / (# of recommended items @k)
    """
    idx = np.argsort(-predict_prob)
    true_label = true_label[idx]
    n_relevant = np.sum(true_label[:k])
    precision = float(n_relevant) / k if k != 0 else 1
    return precision


def recall_at_k(true_label, predict_prob, k):
    """
    Recall@k = (# of recommended items @k that are relevant) / (total # of relevant items)
    """
    idx = np.argsort(-predict_prob)
    true_label = true_label[idx]
    n_relevant = np.sum(true_label[:k])
    recall = 1 if np.sum(true_label) == 0 else float(n_relevant) / np.sum(true_label)
    return recall


def first_hit(true_label, predict_prob):
    idx = np.argsort(-predict_prob)
    true_label = true_label[idx]
    if np.sum(true_label) == 0:
        return true_label.shape[0] + 1
    else:
        return np.nonzero(true_label)[0][0]


def precision_top_k(true_label, predict_prob):
    """
    Calculate precision with setting K = n_positive.
    """
    k = np.sum(true_label)
    idx = np.argsort(-predict_prob)
    true_label = true_label[idx]
    n_relevant = np.sum(true_label[:k])
    precision = float(n_relevant) / k if k != 0 else 1
    return precision


def recall_top_k(true_label, predict_prob):
    """
    Calculate recall with setting K = n_positive.
    """
    k = np.sum(true_label)
    idx = np.argsort(-predict_prob)
    true_label = true_label[idx]
    n_relevant = np.sum(true_label[:k])
    recall = float(n_relevant) / k if k != 0 else 1
    return recall


def cdf_percentage(data, p):
    p = float(p)
    data = sorted(data)
    return data[int(len(data) * p)]


def cdf(data, remove_zero=False):
    if remove_zero:
        data = data[data != 0]
    data_sorted = np.sort(data)
    p = 1. * np.arange(len(data)) / (len(data) - 1)
    return data_sorted, p


# Plot part

def plot_cdfs(Xs, Ys, labels, xlabel, ylabel, save=None, xylimits=None):
    assert len(Xs) == len(Ys) == len(labels)
    fig = plt.figure(figsize=(15, 15))
    if xylimits is not None:
        plt.axis(xylimits)
    else:
        plt.axis([-0.1, 1.1, -0.1, 1.1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    lines = []
    for idx, l in enumerate(labels):
        X = Xs[idx]
        Y = Ys[idx]
        l_, = plt.plot(X, Y, label=l)
        lines.append(l_)
    plt.legend(handles=lines, labels=labels, loc='best')
    plt.show()
    if save is not None:
        fig.savefig(save)


def plot_cdf(X, Y, xlabel, ylabel, save=None, xylimits=None):
    fig = plt.figure(figsize=(10, 10))
    if xylimits is not None:
        plt.axis(xylimits)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(X, Y)
    plt.show()
    if save is not None:
        fig.savefig(save)
