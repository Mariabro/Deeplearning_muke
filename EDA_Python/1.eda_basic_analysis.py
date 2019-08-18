# __author: zxt
# date: 2019/8/15
# 确保一个数除以另外一个数不等于0，会返回一个浮点型的数，另外一种方法是使分母和分子做类型转换
from __future__ import division
import numpy as np
import pandas as pd
from scipy import stats


##0.Read Data##
df = pd.read_csv('./data/train.csv')  # 将数据文件读到内存中
label = df['TARGET']    # 将数据集中TARGET列赋值给label
df = df.drop(['ID', 'TARGET'], axis=1)      # 删除数据集中ID、TARGET两列

##1.Basic Analysis##
# (1)Missing Value
missSet = [np.nan, 9999999999, -999999]     # np.nan是numpy中缺失值的表示，后面两个数是分析得到的缺失值

# (2)Count distinct
len(df.iloc[:, 0].unique())     # df.iloc[:, 0]取第一列，unique()看这一列哪些不同的值，返回一个类似数组，然后len看有多少个

count_un = df.iloc[:, 0:3].apply(lambda x: len(x.unique()))  # df.iloc[:, 0:3]取前三列，apply用于遍历，看看前三列每一列有多少个不同的值

# (3)Zero Value  看看每一列有多少个值为0
np.sum(df.iloc[:, 0] == 0)

count_zero = df.iloc[:, 0:3].apply(lambda x: np.sum(x == 0))

# (4)Mean Value
np.mean(df.iloc[:, 0])   # 没有去除缺失值之前的均值很低

df.iloc[:, 0][~np.in1d(df.iloc[:, 0], missSet)]    # 去除缺失值，np.in1d(df.iloc[:, 0], missSet)找到缺失值的位置，是缺失值返回true，但我们要找的是去除缺失值之后的数据
np.mean(df.iloc[:, 0][~np.in1d(df.iloc[:, 0], missSet)])   # 去除缺失值后进行均值计算

df_mean = df.iloc[:, 0:3].apply(lambda x: np.mean(x[~np.in1d(x, missSet)]))

# (5)Median Value
np.median(df.iloc[:, 0])    # 没有去除缺失值之前

df.iloc[:, 0][~np.in1d(df.iloc[:, 0], missSet)]     # 去除缺失值
np.median(df.iloc[:, 0][~np.in1d(df.iloc[:, 0], missSet)])      # 去除缺失值后进行计算

df_median = df.iloc[:, 0:3].apply(lambda x: np.median(x[~np.in1d(x, missSet)]))

# (6)Mode Value
df_mode = df.iloc[:, 0:3].apply(lambda x: stats.mode(x[~np.in1d(x, missSet)])[0][0])    # mode这个函数返回的是一个数组，对应每一列出现频率最高的数以及它出现的频数，[0][0]表示取的是这个数字

# (7)Mode Percetage
df_mode_count = df.iloc[:, 0:3].apply(lambda x: stats.mode(x[~np.in1d(x, missSet)])[1][0])   # [1][0]表示取的这个众数出现的频数

df_mode_perct = df_mode_count/df.shape[0]       # df.shape[0]确定有多少个样本

# (8)Min Value
np.min(df.iloc[:, 0])

df.iloc[:, 0][~np.in1d(df.iloc[:, 0], missSet)]     # 去除缺失值
np.min(df.iloc[:, 0][~np.in1d(df.iloc[:, 0], missSet)])   # 去除缺失值后进行最小值计算

df_min = df.iloc[:, 0:3].apply(lambda x: np.min(x[~np.in1d(x, missSet)]))

# (9)Max Value
np.max(df.iloc[:, 0])

df.iloc[:, 0][~np.in1d(df.iloc[:, 0], missSet)]     # 去除缺失值
np.max(df.iloc[:, 0][~np.in1d(df.iloc[:, 0], missSet)])   # 去除缺失值后进行最大值计算

df_max = df.iloc[:, 0:3].apply(lambda x: np.max(x[~np.in1d(x, missSet)]))

# (10)quantile value
np.percentile(df.iloc[:, 0], (1, 5, 25, 50, 75, 95, 99))    # 第二个参数是分位点，这个设置和用户定义有关

df.iloc[:, 0][~np.in1d(df.iloc[:, 0], missSet)]     # 去除缺失值
np.percentile(df.iloc[:, 0][~np.in1d(df.iloc[:, 0], missSet)], (1, 5, 25, 50, 75, 95, 99))      # 去除缺失值后进行分位点计算

# 它没法像df那样做一个apply操作，apply返回的是一个数组，那么如何做呢？
json_quantile = {}

# 不用担心循环的效率，因为我们的列不会有很多，虽然样本（行）数会很大，所以一般情况足够用于我们进行数据分析了
for i, name in enumerate(df.iloc[:, 0:3].columns):
    print('the {} columns: {}'.format(i, name))
    json_quantile[name] = np.percentile(df[name][~np.in1d(df[name], missSet)], (1, 5, 25, 50, 75, 95, 99))

df_quantile = pd.DataFrame(json_quantile)[df.iloc[:, 0:3].columns].T    # 为了和之前的统计描述拼接起来，所以需要和之前的结果保持一致，需要做一下转置，但是列名不一致，需要先调用columns把所有的列按顺序发过去之后再转置

# (11)Frequent Value
df.iloc[:, 0].value_counts().iloc[0:5, ]      # value_counts是pandas中dataframe的方法，显示指定特征按照频数由大到小排序，我们一般取前五位频繁出现的值以及它的频数
# 至于选择0:5还是0:10，根据业务定义，一般选取前五位就已经看出一些问题了

# 缺失值不应该存在于EDA中
df.iloc[:, 0][~np.in1d(df.iloc[:, 0], missSet)]     # 去除缺失值
df.iloc[:, 0][~np.in1d(df.iloc[:, 0], missSet)].value_counts()[0:5]     # 去除缺失值后进行频数的统计

# 和分位点的处理方法类似，不能直接用apply
json_fre_name = {}  # 名字
json_fre_count = {}     # 计数


# 如果特征不够5怎么办？剩下的置空。有两个目的：第一，定长，为了和前面的值一致；第二，留一些位置以便更好地拓展
def fill_fre_top_5(x):
    if(len(x)) <= 5:
        new_array = np.full(5, np.nan)
        new_array[0:len(x)] = x
        return new_array


df['ind_var1_0'].value_counts()   # 小于5
df['imp_sal_var16_ult1'].value_counts()    # 大于5

for i, name in enumerate(df[['ind_var1_0', 'imp_sal_var16_ult1']].columns):   # columns取其列名
    # 1.Index Name
    index_name = df[name][~np.in1d(df[name], missSet)].value_counts().iloc[0:5, ].index.values
    # 1.1 If the length of array is less than 5
    index_name = fill_fre_top_5(index_name)

    json_fre_name[name] = index_name

    # 2.Value Count
    values_count = df[name][~np.in1d(df[name], missSet)].value_counts().iloc[0:5, ].values
    # 2.1 If the length of array is less than 5
    values_count = fill_fre_top_5(values_count)

    json_fre_count[name] = values_count

df_fre_name = pd.DataFrame(json_fre_name)[df[['ind_var1_0', 'imp_sal_var16_ult1']].columns].T   # 为了保证格式一致
df_fre_count = pd.DataFrame(json_fre_count)[df[['ind_var1_0', 'imp_sal_var16_ult1']].columns].T

df_fre = pd.concat([df_fre_name, df_fre_count], axis=1)     # concat合并

# (12)Miss Value
np.sum(np.in1d(df.iloc[:, 0], missSet))     # 统计缺失值
df_miss = df.iloc[:, 0:3].apply(lambda x: np.sum(np.in1d(x, missSet)))  # 遍历每一个遍历的缺失值情况，因为返回的是一个值，所以直接用apply遍历
