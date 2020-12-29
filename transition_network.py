# -*- encoding: utf-8 -*-
'''
@File    :   transition_network.py
@Time    :   2020/12/22 20:16:17
@Author  :   DataMagician 
@Version :   1.0
@Contact :   408903228@qq.com
@Desc    :   None
'''

# here put the import lib

import numpy as np
import pandas as pd
import cupy as cp
import os
import jieba

table_path = os.path.expanduser("~/孟哥版/StaticCache/linux_cmd.csv")
table = pd.read_csv(table_path)
data = table[table.columns[4]].to_numpy()


def the_probability_of_function(data):
    """
    '概率'
    """
    return data

def the_transfer_matrix_of_probability_function():
    """
    '概率转移矩阵'
    """
    pass

def prior_to_the_network_function():
    """
    '前向网络'
    """
    pass

def after_to_the_network_function():
    """
    '后向网络'
    """
    pass

variable_the_probability_of = "" #概率

variable_the_transfer_matrix_of_probability = "" #概率转移矩阵

variable_prior_to_the_network = "" #前向网络

variable_after_to_the_network = "" #后向网络


if __name__ == "__main__":
    print(the_probability_of_function(data[0]))