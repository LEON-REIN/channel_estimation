# @.@ coding  : utf-8 ^-^
# @Author     : Leon Rein
# @Time       : 2021-03-09  ~  15:58 
# @File       : 1.generating_original_data.py
# @Software   : PyCharm
# @Notice     : It's a WINDOWS version!
#               只用来生成调制前的原始数据(经过星座映射), 可以用作标签. 若用于调制, 需额外转换为复数.

"""
输入: 无

输出: labels48_test.csv
    一个 CSV 表格文件, 共 1000 行 48 列, 共 48000 个四进制码元.
    每行表示一个 OFDM 符号要调制的 48 个数据码元, 共 1000 个样本.
    调制阶数为 4, 每个数据码元为 {0, 1, 2, 3} 中的一个, 且均匀分布.

    labels48_train.npy
    基本同上, 除了行数为 10000

"""


import numpy as np


np.random.seed(1)  # To guarantee the same random result every time
# Uniform generation of symbols
original_symbols_test = np.random.randint(low=0, high=4, size=(1000, 48), dtype=np.int)  # 0, 1, 2, 3
# Save data set
# np.savetxt("./data_sets/labels48_test.csv", original_symbols, delimiter=',', fmt="%d")

np.random.seed(777)
original_symbols_train = np.random.randint(low=0, high=4, size=(10000, 48), dtype=np.int)
# np.save("./data_sets/labels48_train.npy", original_symbols_train)
# original_symbols_train = np.load("./data_sets/labels48_train.npy")  # load the data set
