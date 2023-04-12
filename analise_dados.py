
from statistics import mean, median
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataall = pd.read_csv('nanocav_datafull.csv', index_col=False)


# print(dataall.head())
a = dataall.iloc[:, 1].values
rx = dataall.iloc[:, 2].values*350
ry = dataall.iloc[:, 3].values*350
a_min = dataall.iloc[:, 4].values
rx_min = dataall.iloc[:, 5].values*350
ry_min = dataall.iloc[:, 6].values*350
del_a = dataall.iloc[:, 7].values
del_rx = dataall.iloc[:, 8].values
del_ry = dataall.iloc[:, 9].values
del_lc = dataall.iloc[:, 10].values

#-- Outputs
v = dataall.iloc[:, 11].values
q = dataall.iloc[:, 12].values
l_0 = dataall.iloc[:, 13].values
f_p = dataall.iloc[:, 14].values

# a forma 100x
# a_min forma 100x - formato log
# del_a forma 10x
# del_rx forma 10x
# del_ry forma 10x

# -- v 1e-19 formato log
# -- q 5e3 formato log
# -- l_0 9e-7 formato dist
# -- f_p 50

# plt.hist(f_p, 20)
# plt.title('F_p')
# plt.show()

correl = []
head = ['a', 'a_min', 'del_a', 'rx', 'rx_min',
        'del_rx', 'ry', 'ry_min', 'del_ry', 'del_lc']
for cols in head:
    correlacao = dataall[cols].corr(dataall['Q'])
    correl.append(correlacao)
#     print(correlacao)
print(np.mean(q))
