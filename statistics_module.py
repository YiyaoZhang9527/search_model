# -*- encoding: utf-8 -*-
'''
@File    :   statistical.py
@Time    :   2020/10/28 00:24:33
@Author  :   DataMagician 
@Version :   1.0
@Contact :   408903228@qq.com
'''

# here put the import lib

import numpy as np
import numexpr as en
from itertools import product
from data_structure_module import check_ndarray_function


def sort_by_dictionary_value_function(dictionary):
    '''
    按字典值排序
    '''
    return sorted(dictionary.items(), key=lambda d: d[1], reverse=True)


def probability_optimization_function(prob, optimization=False):
    '''
    概率表达方式优化模块
    '''
    if optimization == False:
        prob = prob
    elif optimization == 'log':
        prob = np.log(prob)
    elif optimization == '-log':
        prob = -np.log(prob)
    return prob


def return_type_function(sorted_data, return_type):
    '''
    选择返回的数据结构
    '''
    if return_type == "dict":
        return dict(zip(sorted_data[:, 0], sorted_data[:, -1].astype(float)))
    if return_type == 'tuple':
        return sorted_data[:, 0], sorted_data[:, -1].astype(float)
    return sorted_data


def frequency_function(data, return_type="dict"):
    '''
    频率计算并排序
    '''
    init_data = check_ndarray_function(data)
    unique, counter = np.unique(init_data, return_counts=True)
    sorted_ = np.array(sort_by_dictionary_value_function(dict(zip(unique, counter))))
    return return_type_function(sorted_, return_type)


def probaitily_function(x, return_type="dict", optimization=False):
    '''
    概率计算
    '''
    init_x = check_ndarray_function(x)
    lenght = init_x.size
    elements, counter = np.unique(init_x, return_counts=True)
    prob = counter / lenght
    sorted_ = sorted_ = np.array(
        sort_by_dictionary_value_function(dict(zip(elements, probability_optimization_function(prob, optimization)))))
    return return_type_function(sorted_, return_type)


def sigmoid_function(x):
    '''sigmoid函数'''
    return 1 / (1 + np.exp(x * -1))


def combinationL(loop_val):
    return np.array(list({i for i in product(*loop_val)}))


def NumpyProb(X, symbol, x):
    n = X.size
    Lambda = "{}{}{}".format("X", symbol, "x")
    expr = en.evaluate(Lambda)
    return (expr).dot(np.ones(n)) / n


def NumpyJointProb(X, Y):
    init_XY = np.c_[X, Y]
    distionXY = combinationL(init_XY.T)
    m, n = init_XY.shape
    dm, dn = distionXY.shape
    if dm == 1:
        distionXY = np.repeat(distionXY, dn, axis=0).T
        return np.array([((init_XY.T == xy).dot(np.ones(n)) == n).dot(np.ones(m)) / m for xy in distionXY])
    elif dm > 1:
        return np.array([((init_XY == xy).dot(np.ones(n)) == n).dot(np.ones(m)) / m for xy in distionXY])


def NumpyEntropy(X):
    NE = 0
    for x in np.unique(X):
        PX = NumpyProb(X, '==', x)
        NE += (- PX * np.log2(PX))
    return NE


def NumpyJointEntropy(X, Y):
    PXY = NumpyJointProb(X, Y)
    return (-PXY * np.log2(PXY)).sum()



def euclidean(x, y):
    '''
    欧几里得度量
    （euclidean metric）（也称欧氏距离）是一个通常采用的距离定义，指在m维空间中两个点之间的真实距离，或者向量的自然长度（即该点到原点的距离）。在二维和三维空间中的欧氏距离就是两点之间的实际距离。
    $$ d(x,y)= \sqrt{\sum_i{(x_i-y_i)^2}} $$
    Args:
        x:
        y:

    Returns:

    '''
    return np.sqrt(np.sum((x - y)**2))


def manhattan(x, y):
    '''
    曼哈顿距离
    想象你在曼哈顿要从一个十字路口开车到另外一个十字路口，驾驶距离是两点间的直线距离吗？显然不是，除非你能穿越大楼。实际驾驶距离就是这个“曼哈顿距离”。而这也是曼哈顿距离名称的来源， 曼哈顿距离也称为城市街区距离(City Block distance)。
    $$ d(x,y) = \sum_i{|x_i-y_i|} $$
    Args:
        x:
        y:

    Returns:

    '''
    return np.sum(np.abs(x - y))




def chebyshev(x, y):
    '''
    切比雪夫距离
    在数学中，切比雪夫距离（Chebyshev distance）或是L∞度量，是向量空间中的一种度量，二个点之间的距离定义是其各坐标数值差绝对值的最大值。以数学的观点来看，切比雪夫距离是由一致范数（uniform norm）（或称为上确界范数）所衍生的度量，也是超凸度量（injective metric space）的一种。
    $$ d(x,y) = max_i{|x-y|}$$
    Args:
        x:
        y:

    Returns:

    '''

    return np.max(np.abs(x - y))




def minkowski(x, y, p):
    '''
    闵可夫斯基距离
    闵氏空间指狭义相对论中由一个时间维和三个空间维组成的时空，为俄裔德国数学家闵可夫斯基(H.Minkowski,1864-1909)最先表述。他的平坦空间（即假设没有重力，曲率为零的空间）的概念以及表示为特殊距离量的几何学是与狭义相对论的要求相一致的。闵可夫斯基空间不同于牛顿力学的平坦空间。

    $$ d(x,y)={(\sum{|x_i,y_i|^p})}^{1/p} $$
    Args:
        x:
        y:
        p:

    Returns:

    '''

    return np.sum(np.abs(x - y) ** p) ** (1 / p)


def hamming(x, y):
    '''
    汉明距离
    汉明距离是使用在数据传输差错控制编码里面的，汉明距离是一个概念，它表示两个（相同长度）字对应位不同的数量，我们以$d(x,y)$表示两个字$x,y$之间的汉明距离。对两个字符串进行异或运算，并统计结果为1的个数，那么这个数就是汉明距离.

    $$ d(x,y)= 1/N \sum_i{x_i \neq y_i} $$
    Args:
        x:
        y:

    Returns:

    '''
    return sum(x!=y)/len(x)

def jaccard_function(x,y):
    '''
    亚卡尔系数代表的是集合的相似程度
    x
    Args:
        x:
        y:

    Returns:

    '''
    return np.intersect1d(x,y).shape[0]/np.union1d(x,y).shape[0]

def cosangle_function(x,y):
    '''

    Args:
        x:
        y:

    Returns:

    '''
    cosl =x.dot(y)/(np.linalg.norm(x) * np.linalg.norm(y))
    pi = np.pi
    cos = cosl*2*pi
    radian = cos*(pi/180)
    return {'cosangle':cosl,'cos':cos,'radian':radian}



if __name__ == "__main__":
    case = np.array(['查询', 'GPU', '或者', '检查', '时间', '存储', '文件', '开房', '检查', '时间', '存储', '文件', '开房', '扫描', '目标',
                     '网络工具', '是', '内', '上', '输入', '，', '了', '信息', '数据', '哪些', '运行',
                     'UDP', '设备', '识别', '端口', '留下', ' ', '集', '本', '日志', '哪个', '到',
                     '酒店', '在线', '项目', '系统', '查看', '/', '历史', '机', '的', '网段', '》',
                     '端口扫描', '在', '服务器', '很少', '隐藏', '激活', '启动', '某台', '存活', '有',
                     '大型项目', '：', '本地', '公网', '开放', '《', '开机', 'ip', '只', '位置', '同步',
                     '操作系统', '华为', '所在', '上传', '盘', '主机'])
    print(frequency_function(case, return_type="tuple"))
    # print(probaitily(case,return_type='dict',optimization='-log'))#,optimization='-log'))
