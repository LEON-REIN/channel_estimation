# @.@ coding  : utf-8 ^-^
# @Author     : Leon Rein
# @Time       : 2021-03-09  ~  18:50 
# @File       : 2.parallel_data_64.py
# @Software   : PyCharm
# @Notice     : It's a WINDOWS version!
#               从原始码元中间插入 DC 置零, 插入前 6 个后 5 个置零的虚拟子载波作为保护,
#               在下标 -21, -7, 7, 21 处插入导频 3 (当作 1 + i)

"""
输入: labels48.csv

输出: labels64.csv
    一个 CSV 表格文件, 共 1000 行 64 列, 每一行如下
    下标 [0:6] 共 6 个为置零子载波, 为 0;
    下标 [6:11] 共 5 个为数据位置;
    下标 [11] 是第一个导频, 为 3;
    下标 [12:25] 共 13 个为数据位置;
    下标 [25] 是第二个导频, 为 3;
    下标 [26:32] 共 6 个为数据位置;
    下标 [32] 为中间直流子载波置零, 为 0;
    下标 [33:39] 共 6 个为数据位置;
    下标 [39] 是第三个导频, 为 3;
    下标 [40:53] 共 13 个为数据位置;
    下标 [53] 是第四个导频, 为 3;
    下标 [54:59] 共 5 个为数据位置;
    下标 [59:64] 共 5 个为置零子载波, 为 0.
"""

import numpy as np

original_48 = np.loadtxt("./data_sets/labels48.csv", delimiter=",").astype(np.int)

original_64 = np.zeros(shape=(1000, 64), dtype=np.int)
original_64[:, 6:11] = original_48[:, 0:5]
original_64[:, 11] = 3
original_64[:, 12:25] = original_48[:, 5:18]
original_64[:, 25] = 3
original_64[:, 26:32] = original_48[:, 18:24]
original_64[:, 33:39] = original_48[:, 24:30]
original_64[:, 39] = 3
original_64[:, 40:53] = original_48[:, 30:43]
original_64[:, 53] = 3
original_64[:, 54:59] = original_48[:, 43:48]

# Save data set
# np.savetxt("./data_sets/labels64.csv", original_64, delimiter=',', fmt="%d")


# Transform original_64 into onehot vector. ONLY for 4th order modulation!!
# 0->[1 0 0 0], 1->[0 1 0 0], 2->[0 0 1 0], 3->[0 0 0 1]
def one_hot_mapping(x, num_classes=4):
    mapping_list = np.eye(num_classes)
    return mapping_list[x]


original_64_onehot = np.array([one_hot_mapping(i) for i in original_64])
original_64_onehot = original_64_onehot.reshape(original_64.shape[0], -1)
# Save the result
# np.save("./data_sets/labels64_onehot.npy", original_64_onehot)
# original_64_onehot = np.load("./data_sets/labels64_onehot.npy")  # load the data set











