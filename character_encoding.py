# -*- encoding: utf-8 -*-
'''
@File    :   character_encoding.py
@Time    :   2020/10/28 13:55:07
@Author  :   DataMagician
@Version :   1.0
@Contact :   408903228@qq.com
'''

# here put the import lib
import os
import sys
base_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(base_dir)
from data_preprocessing_module import word_punct_tokenizer_for_chinese_function
from statistics_module import frequency_function
from numpy import ones, array, zeros
from pandas import DataFrame
from data_structure_module import deeping_flatten_function




def character_encoding_function(articles, filter_stop_words=True):
    '''
    对文本列表特征编码
    '''
    tokenization_for_each_articles = word_punct_tokenizer_for_chinese_function(articles, filter_stop_words)  # 分词每段文章
    flatten_ = deeping_flatten_function((word for num, word in tokenization_for_each_articles.items()))  # 将所有文章单词列表展开到一纬
    total_number_of_words = len(flatten_)  # 所有文章单词的数量的总和
    distances, counter = frequency_function(flatten_, return_type="tuple")  # 求每个单词在所有文章中出现的频率
    feature_dictionary_editor = {key: value for key, value in zip(distances, counter / total_number_of_words)}

    distance_lenght = counter.size  # 去重侯单词的数量
    counter_coding = []  # zeros((distance_lenght,distance_lenght))
    one_vector = ones(distance_lenght)  # 全是1的向量
    onehots_encoding_for_each_article = dict()  # 初始化返回列表

    time_leve0 = -1
    counter_word = 0  # 初始化单词计数
    vocabulary = set()
    for No, words in tokenization_for_each_articles.items():
        time_leve0 += 1
        time_leve1 = -1
        init_onehot_mat = zeros((distance_lenght, distance_lenght))  # 单文onehot矩阵
        for word in words:  # 循环计算所有单词
            time_leve1 += 1
            init_onehot_dictionary = {key: value for key, value in
                                      zip(distances, zeros(distance_lenght))}  # 初始化onehot字典
            init_onehot_dictionary[word] = 1  # 将对应单词的字典编码改为1
            init_onehot_vector = array([one_hot_code for word_leve1, one_hot_code in
                                        init_onehot_dictionary.items()])  # 整理每个单词的onehot字典成为onehot向量
            init_onehot_mat[time_leve1] = init_onehot_vector  # 将每篇文章的onehot矩阵对应的单词行修改为当前循环单词的onehot向量
            vocabulary.add(word)
        counter_word += init_onehot_mat.sum()  # 计算每篇文章单词数
        onehots_encoding_for_each_article.update({No: init_onehot_mat})  # 将每篇文章的onehot向量于文章序号对应保存为dict结构
        counter_coding.append(init_onehot_mat.T.dot(one_vector))  # 计算每篇文章词向量
    counter_coding = array(counter_coding)  # 将每一行的对应单词计数转为矩阵表达
    return {"onehots_encoding_for_each_article": onehots_encoding_for_each_article
        , "counter_vectors": DataFrame(counter_coding, columns=distances)
        , "probabilistic_feature_of_words_in_each_article ": DataFrame(counter_coding / total_number_of_words, columns=distances)
        , "vocabulary": vocabulary
        , "probabilistic_feature_dictionary": feature_dictionary_editor
        , "probabilistic_feature_vectors": array([value for key, value in feature_dictionary_editor.items()])
        , "counter": counter_word == total_number_of_words and counter_word or "字数不符"}


if __name__ == "__main__":
    instructions_text = ['扫描本机所在网段上有哪些主机是存活的',
                         '端口扫描：输入目标主机ip，扫描某台主机开放了哪些端口',
                         '隐藏扫描，输入目标主机ip，只在目标主机上留下很少的日志信息',
                         'UDP端口扫描：输入目标主机ip，扫描目标主机开放了哪些UDP端口',
                         '操作系统识别：输入目标主机ip，查询是哪个系统',
                         '上传或者同步大型项目文件到服务器',
                         '检查本机网段内ip',
                         '查看本机网段内 激活/在线 的设备',
                         '查询本地公网ip',
                         '上传《网络工具》项目到GPU服务器',
                         '上传《网络工具》项目到华为服务器',
                         '查询系统运行时间',
                         '查询系统开机时间',
                         '查询系统历史启动时间',
                         '存储盘的位置',
                         '酒店开房数据集的位置']
    #article_list = instructions_text
    #print(character_encoding_function(article_list,filter_stop_words=True))
