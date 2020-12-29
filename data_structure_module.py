# -*- encoding: utf-8 -*-
'''
@File    :   data_structure.py
@Time    :   2020/10/27 23:35:36
@Author  :   DataMagician 
@Version :   1.0
@Contact :   408903228@qq.com
'''

# here put the import lib
from collections import Iterable
import numpy as np


def two_dimensional_flatten_function(mylist):
    '''
    二维平展
    '''
    return [i for j in range(len(mylist)) for i in mylist[j]]


def spread_function(arg):
    '''
    广播函数
    Args:
        arg:

    Returns:

    '''
    ret = []
    for i in arg:
        if isinstance(i, (list, tuple, set)):
            ret.extend(i)
        else:
            ret.append(i)
    return ret


def deeping_flatten_function(mylist):
    '''
    深度平展函数
    Args:
        mylist:

    Returns:

    '''
    result = []
    result.extend(
        spread_function(
            list(map(lambda x: deeping_flatten_function(x) if type(x) in (list, tuple, set) else x, mylist))))
    return result


def data_structure_normalization_function(iter_) -> (list, set, tuple):
    '''
    数据结构归一化到ndarray一纬
    '''
    if isinstance(iter_, Iterable):
        if isinstance(iter_, (list, set, tuple)):
            List_iter = deeping_flatten_function(iter_)
        elif isinstance(iter_, (np.ndarray)):
            List_iter = np.ravel(iter_)
        else:
            return "this function only accepts ndarray, list, tuple, set"
        return np.array(List_iter)
    else:
        return "is not an iteratable object, please check"


def check_ndarray_function(data):
    '''
    检查数据结构有没有归一化到ndarray一纬
    '''
    if isinstance(data, np.ndarray):
        if len(data.shape) == 1:
            return data
        else:
            return data.ravel()
    else:
        return np.array(deeping_flatten_function(data))
