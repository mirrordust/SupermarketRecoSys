# encoding=utf-8
import datetime

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, roc_auc_score


def backward_date_slots(base_datetime, datedelta_slots):
    date_slots = []
    for slot in datedelta_slots:
        dates = [base_datetime - datetime.timedelta(days=i) for i in slot]
        date_slots.append(dates)
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
    [np.array([1]), np.array([2]), np.array([3]), np.arange(3) + 1, np.arange(5) + 1, np.arange(7) + 1,
     np.arange(10) + 1, np.arange(14) + 1, np.arange(21) + 1, np.arange(28) + 1, np.arange(40) + 1, np.arange(50) + 1,
     np.arange(59) + 1])


# 构建用户特征
def features_cust(df, predict_first_date, datedelta_slots):
    cust_behavs = []
    df_cust_features = df.drop_duplicates(subset='vipno', keep='first')[['vipno']].reset_index(drop=True)
    date_slots = backward_date_slots(predict_first_date, datedelta_slots)
    last_date_slot = backward_date_slots(predict_first_date, [datedelta_slots[-1]])
    for cust_ in df_cust_features.values:
        cust_vipno = cust_[0]
        df_cust = df[df['vipno'] == cust_vipno]
        # 不同时间段内购买商品数
        ds = map(get_date, df_cust.sldat)
        # feature vector 1
        purch_count = [np.sum(np.in1d(ds, dates)) for dates in date_slots]
        # 前59天有购买行为的日期
        day_purch_or_not = np.in1d(last_date_slot, ds)
        day_purch_or_not = day_purch_or_not.astype(int)  # feature verctor 2
        # 最后一次购买的日期距预测日期的天数
        last_purch_day = max(ds)
        min_daydelta = abs((last_purch_day - predict_first_date).days)  # feature vector 3
        # 合并(第一列为vipno)
        behav = np.concatenate((cust_, purch_count, day_purch_or_not, [min_daydelta]))
        cust_behavs.append(behav)
    return np.array(cust_behavs)


# 构建商品特征
def features_goods(df, predict_first_date, datedelta_slots):
    goods_behavs = []
    df_goods_features = df.drop_duplicates(subset='pluno', keep='first')[['pluno', 'bndno', 'dptno3']].reset_index(
        drop=True)
    date_slots = backward_date_slots(predict_first_date, datedelta_slots)
    print '# goods:', df_goods_features.shape[0]
    i = 0
    for goods_ in df_goods_features.values:
        if i == int(0.33 * df_goods_features.shape[0]):
            print '33% finish'
        elif i == int(0.66 * df_goods_features.shape[0]):
            print '66% finish'
        i = i + 1
        # 商品被购买次数
        pluno = goods_[0]
        df_goods = df[df['pluno'] == pluno]
        ds1 = map(get_date, df_goods.sldat)
        # feature vector 1
        purch_goods_count = [np.sum(np.in1d(ds1, dates)) for dates in date_slots]
        # 商品种类被购买次数
        dptno3 = goods_[2]
        df_cates = df[df['dptno3'] == dptno3]
        ds2 = map(get_date, df_cates.sldat)
        # feature vector 2
        purch_cates_count = [np.sum(np.in1d(ds2, dates)) for dates in date_slots]
        # 合并(第一列为pluno)
        behav = np.concatenate((goods_, purch_goods_count, purch_cates_count))
        goods_behavs.append(behav)
    return np.array(goods_behavs)


# 构建用户-商品特征
def features_cust_goods(df, predict_first_date, datedelta_slots):
    custgoods_behavs = []
    df_custgoods_pairs = df.drop_duplicates(subset=['vipno', 'pluno'], keep='first')[
        ['vipno', 'pluno', 'dptno3']].reset_index(drop=True)
    date_slots = backward_date_slots(predict_first_date, datedelta_slots)
    print '# df_custgoods_pairs: ', df_custgoods_pairs.shape[0]
    i = 0
    for custgoods_ in df_custgoods_pairs.values:
        if i == int(0.25 * df_custgoods_pairs.shape[0]):
            print '25% finish'
        elif i == int(0.5 * df_custgoods_pairs.shape[0]):
            print '50% finish'
        elif i == int(0.75 * df_custgoods_pairs.shape[0]):
            print '75% finish'
        i = i + 1
        vipno_ = custgoods_[0]
        pluno_ = custgoods_[1]
        dptno3_ = custgoods_[2]
        # 商品被用户购买次数
        df_cust_goods = df[(df['vipno'] == vipno_) & (df['pluno'] == pluno_)]
        ds1 = map(get_date, df_cust_goods.sldat)
        # feature vector 1
        purch_goods_count = [np.sum(np.in1d(ds1, dates)) for dates in date_slots]
        # 商品种类被用户购买次数
        df_cust_cates = df[(df['vipno'] == vipno_) & (df['dptno3'] == dptno3_)]
        ds2 = map(get_date, df_cust_cates.sldat)
        # feature vector 2
        purch_cates_count = [np.sum(np.in1d(ds2, dates)) for dates in date_slots]
        # 合并
        behav = np.concatenate(([vipno_, pluno_], purch_goods_count, purch_cates_count))
        custgoods_behavs.append(behav)
    return np.array(custgoods_behavs)


def extract_features(df, predict_first_date, datedelta_slots, loadfromfile=False):
    timestr = '{}_{}_{}'.format(predict_first_date.year, predict_first_date.month, predict_first_date.day)
    # 前两个月的数据提取特征
    if loadfromfile:
        cust_f = np.loadtxt('cust_f_{}.csv'.format(timestr), delimiter=',')
        goods_f = np.loadtxt('goods_f_{}.csv'.format(timestr), delimiter=',')
        custgoods_f = np.loadtxt('custgoods_f_{}.csv'.format(timestr), delimiter=',')
    else:
        cust_f = features_cust(df, predict_first_date, datedelta_slots)
        print 'cust features finish'
        np.savetxt('cust_f_{}.csv'.format(timestr), cust_f, delimiter=',')
        goods_f = features_goods(df, predict_first_date, datedelta_slots)
        print 'goods features finish'
        np.savetxt('goods_f_{}.csv'.format(timestr), goods_f, delimiter=',')
        custgoods_f = features_cust_goods(df, predict_first_date, datedelta_slots)
        print 'custgoods features finish'
        np.savetxt('custgoods_f_{}.csv'.format(timestr), custgoods_f, delimiter=',')
    cust_df = pd.DataFrame(data=cust_f)
    cust_df.columns = ['vipno'] + ['cust_' + str(i) for i in range(1, cust_df.shape[1])]
    goods_df = pd.DataFrame(data=goods_f)
    goods_df.columns = ['pluno'] + ['goods_' + str(i) for i in range(1, goods_df.shape[1])]
    custgoods_df = pd.DataFrame(data=custgoods_f)
    custgoods_df.columns = ['vipno', 'pluno'] + ['custgoods_' + str(i) for i in range(2, custgoods_df.shape[1])]
    training_data = custgoods_df.iloc[:, 0:2]
    merge1 = pd.merge(training_data, cust_df, sort=False, on='vipno', how='left', copy=False)
    merge2 = pd.merge(merge1, goods_df, sort=False, on='pluno', how='left', copy=False)
    merge2 = merge2.fillna(-1)
    merge3 = pd.merge(merge2, custgoods_df, sort=False, on=['vipno', 'pluno'], how='left', copy=False)
    # 预实验 取 20*20的数据
    # customers = customers[np.random.randint(0, customers.shape[0], 20)]
    # goods = goods[np.random.randint(0, goods.shape[0], 20)]
    return merge3


def get_report(proba, true_label):
    idx = np.argsort(-proba)
    proba = proba[idx]
    true_label = true_label[idx]
    auc = roc_auc_score(true_label, proba)
    print 'Total\t%d\nbuyer\t%d\nRate\t%5.4f\nAUC\t%5.4f' % (
        true_label.size, sum(true_label), float(sum(true_label)) / true_label.size, auc)
    valid_size = [100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000]
    print 'Top\tbuyer\tRecall\tPrecision\t'
    predict = np.zeros(true_label.size)
    for size in valid_size:
        predict[0:size] = 1
        recall = recall_score(true_label, predict, average='binary')
        precision = precision_score(true_label, predict, average='binary')
        label = true_label[:size]
        buyer = len(np.where(label == 1)[0])
        print '%d\t%d\t%5.4f\t%5.4f' % (size, buyer, recall, precision)
