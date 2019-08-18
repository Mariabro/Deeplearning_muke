# __author: zxt
# date: 2019/8/16
# -*- coding:utf-8 -*-
import pandas as pd
from wrap_up import *
from wrap_up_cal_time import eda_analysis_cal_time


# 0.Read Data
df = pd.read_csv('./data/train.csv')
label = df['TARGET']
df = df.drop(['ID', 'TARGET'], axis=1)

# 1.EDA
df_eda_summary = eda_analysis(missSet=[np.nan, 9999999999, -999999], df=df.iloc[:, 0:3])

# 可以看到求众数的时间较长，那么在后期优化的时候可以把重点放在这里
# 优化的方法：把没有用的代码注释掉；或者寻求更好的方法去替代原有方法
df_eda_summary = eda_analysis_cal_time(missSet=[np.nan, 99999999999, -999999], df=df.iloc[:, 0:3])

# 2.Calculating Running time
import timeit

start = timeit.default_timer()
df_eda_summary = eda_analysis(missSet=[np.nan, 9999999999, -999999], df=df.iloc[:, 0:3])
print('EDA Running Time: {0:.2f} seconds'.format(timeit.default_timer() - start))


