# __author: zxt
# date: 2019/8/16
# -*-coding:utf-8 -*-
from __future__ import division
import pandas as pd
import numpy as np
from scipy import stats


def fill_fre_top_5(x):
    if(len(x)) <= 5:
        new_array = np.full(5, np.nan)
        new_array[0:len(x)] = x
        return new_array


def eda_analysis(missSet=[np.nan,9999999999,-999999], df=None):
    # 1.Count
    count_un = df.apply(lambda x: len(x.unique()))
    count_un = count_un.to_frame('count')   # to_frame转成dataframe类型，并取名

    # 2.Count Zero
    count_zero = df.apply(lambda x: np.sum(x == 0))
    count_zero = count_zero.to_frame('count_zero')

    # 3.Mean
    df_mean = df.apply(lambda x: np.mean(x[~np.in1d(x, missSet)]))
    df_mean = df_mean.to_frame('mean')

    # 4.Median
    df_median = df.apply(lambda x: np.median(x[~np.in1d(x, missSet)]))
    df_median = df_median.to_frame('median')

    # 5.Mode
    df_mode = df.apply(lambda x: stats.mode(x[~np.in1d(x, missSet)])[0][0])     # 注意括号的位置，在这里犯了错
    df_mode = df_mode.to_frame('mode')

    # 6.Mode Count
    df_mode_count = df.apply(lambda x: stats.mode(x[~np.in1d(x, missSet)])[1][0])
    df_mode_count = df_mode_count.to_frame('mode_count')

    # 6.1 Mode Percentage
    df_mode_perct = df_mode_count / df.shape[0]
    df_mode_perct.columns = ['mode_perct']

    # 7.Min
    df_min = df.apply(lambda x: np.min(x[~np.in1d(x, missSet)]))
    df_min = df_min.to_frame('min')

    # 8.Max
    df_max = df.apply(lambda x: np.max(x[~np.in1d(x, missSet)]))
    df_max = df_max.to_frame('max')

    # 9.Quantile
    json_quantile = {}

    for i, name in enumerate(df.columns):
        json_quantile[name] = np.percentile(df[name][~np.in1d(df[name], missSet)], (1, 5, 25, 50, 75, 95, 99))

    df_quantile = pd.DataFrame(json_quantile)[df.columns].T
    df_quantile.columns = ['quan01', 'quan05', 'quan25', 'quan50', 'quan75', 'quan95', 'quan99']

    # 10.Frequence
    json_fre_name = {}
    json_fre_count = {}

    for i, name in enumerate(df.columns):
        # 1.Index Name
        index_name = df[name][~np.in1d(df[name], missSet)].value_counts().iloc[0:5, ].index.values
        # If the length of array is less than 5
        index_name = fill_fre_top_5(index_name)

        json_fre_name[name] = index_name

        # 2.Value Count
        values_count = df[name][~np.in1d(df[name], missSet)].value_counts().iloc[0:5, ].values
        # If the length of array is less than 5
        values_count = fill_fre_top_5(values_count)

        json_fre_count[name] = values_count

    df_fre_name = pd.DataFrame(json_fre_name)[df.columns].T
    df_fre_count = pd.DataFrame(json_fre_count)[df.columns].T

    df_fre = pd.concat([df_fre_name, df_fre_count], axis=1)
    df_fre.columns = ['value1', 'value2', 'value3', 'value4', 'value5', 'freq1', 'freq2', 'freq3', 'freq4', 'freq5']

    # 11.Miss Value Count
    df_miss = df.apply(lambda x: np.sum(np.in1d(x, missSet)))
    df_miss = df_miss.to_frame('freq_miss')

    # 12.Combine All Information
    df_eda_summary = pd.concat(
        [count_un, count_zero, df_mean, df_median, df_mode,
         df_mode_count, df_mode_perct, df_min, df_max, df_fre,
         df_miss], axis=1
    )
    # 左边是特征，上边是有多少统计描述，就拼多少

    return df_eda_summary



