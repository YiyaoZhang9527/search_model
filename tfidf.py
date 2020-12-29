# -*- encoding: utf-8 -*-
'''
@File    :   tfidf.py
@Time    :   2020/10/05 22:09:15
@Author  :   DataMagician 
@Version :   1.0
@Contact :   408903228@qq.com
'''

# here put the import lib

import numpy as np
from numpy import argwhere, zeros, ndarray, array, log, ones
from data_preprocessing_module import word_punct_tokenizer_for_chinese_function
from tqdm import tqdm

need_type = list, tuple, ndarray, set


# TODO : 分步TF算法
def tf_function(original_list: (need_type), word_vector) -> need_type:
    '''

    Args:
        paper_words: 文章列表
        word_vector: 词汇表

    Returns:

    '''
    m = len(original_list)
    init_TF = zeros(m)
    for word in original_list:
        if word in word_vector:
            index_ = argwhere(word_vector == word)[0][0]
            init_TF[index_] += 1
    return init_TF


# TODO ：分步IDF算法
def idf_function(paper_words_list, word_vector):
    '''

    Args:
        paper_words_list: 文章列表
        word_vector: 词汇表

    Returns:

    '''
    m = word_vector.size
    init_IDF = zeros(m)
    N = paper_words_list.shape
    n = -1
    for word in word_vector:
        n += 1
        for paper_arr in paper_words_list:
            if word in paper_arr:
                init_IDF[n] += 1
    return np.log(N / (init_IDF + 1))


# TODO : 一次训练的整个训练TFIDF词向量
def data_preprocessing_for_tfidf_function(original: (list, tuple, ndarray, set)
                                          , filter_stop_words=True) -> (bool, list, tuple, ndarray, set):
    '''

    Args:
        paper_words_list:
        word_vector:

    Returns:

    '''
    word_punct_tokenizer = word_punct_tokenizer_for_chinese_function(original
                                                                     , filter_stop_words=filter_stop_words)
    vocabulary = []
    for No, paper_tokens in word_punct_tokenizer.items():
        vocabulary += paper_tokens
    empty_words_dictionay = dict(zip(vocabulary, zeros(len(vocabulary))))
    return {"word_punct_tokenizer": word_punct_tokenizer
        , "vocabulary_set": set(vocabulary)
        , "empty_words_dictionay": empty_words_dictionay}


def TFIDF_function(original_list: (list, ndarray, set, tuple)
                   , filter_stop_words=True):
    '''
    Args:
        word_punct_tokens: 分词后的文章列表
        vocabulary: 词汇表

    Returns: tf 矩阵 横向量为词汇表

    TF-IDF 是一个统计方法，用来评估某个词语对于一个文件集或文档库中
    的其中一份文件的重要程度。TF-IDF 实际上是两个词组 Term Frequency
    和 Inverse Document Frequency 的总称，两者缩写为 TF 和 IDF，
    分别代表了词频和逆向文档频率。词频 TF 计算了一个单词在文档中出现的次数
    ，它认为一个单词的重要性和它在文档中出现的次数呈正比。逆向文档频率 IDF
    ，是指一个单词在文档中的区分度。它认为一个单词出现在的文档数越少，
    就越能通过这个单词把该文档和其他文档区分开。IDF 越大就代表该单词的区分
    度越大。所以 TF-IDF 实际上是词频 TF 和逆向文档频率 IDF 的乘积
    。这样我们倾向于找到 TF 和 IDF 取值都高的单词作为区分，
    即这个单词在一个文档中出现的次数多，同时又很少出现在其他文档中。
    这样的单词适合用于分类。

    $$ 词频TF = 单词出现的次数/该文档的总单词数 $$
    $$ 逆向文档的频率IDF = log(文档总数/该单词出现的文档数+1)
    $$ 词频TF * 逆向文档的频率IDF

    '''
    word_punct_tokenizer = data_preprocessing_for_tfidf_function(original_list
                                                                 , filter_stop_words=filter_stop_words)
    word_punct_tokens, vocabulary = word_punct_tokenizer["word_punct_tokenizer"] \
        , word_punct_tokenizer["vocabulary_set"]
    m, n = len(word_punct_tokens), len(vocabulary)
    init_TF = zeros((m, n))
    init_IDF = zeros(n)
    init_counter_words_for_each_document = zeros(m)
    for No, paper_tokens in tqdm(word_punct_tokens.items(),"tfidf训练"):
        vocabulary_of_each_document = len(paper_tokens)
        init_TF += array([paper_tokens.count(word) for word in vocabulary])
        init_IDF += array([word in paper_tokens and 1 or 0 for word in vocabulary])
        init_counter_words_for_each_document[No] = vocabulary_of_each_document

    TF = (init_TF.T / init_counter_words_for_each_document).T

    IDF = log(m / (init_IDF + 1))

    return {"TF-IDF": TF * IDF, "TF": TF, "IDF": IDF, "init_TF": init_TF[0], "words_counter": init_IDF.dot(ones(n)),
            "document_count": m,
            "vocabulary_from_TF-IDF": vocabulary}  # ,"counter_vector":init_IDF}


# TODO : TFIDF预训练模型部分
def trained_TF_model_function(words_list
                              , init_TF
                              , vocabulary):
    '''

    Args: 词频TF = 单词出现的次数/该文档的总单词数
        words_list: 要续上训练的文本分词列表
        vocabulary: 词列表

    Returns: 单词出现的次数/该文档的总单词数

    '''
    m = len(words_list)
    return (array([words_list.count(word) for word in vocabulary]) + init_TF) / m


def trained_IDF_model_function(document_count
                               , words_list
                               , vocabulary
                               , IDF_src):
    '''

    Args: 逆向文档的频率IDF = log(文档总数/该单词出现的文档数+1)
        document_count: 历史文档总数
        words_list: 要续上的文本分词列表
        vocabulary: 词列表
        IDF_src: 原本的IDF模型

    Returns:

    '''
    init_IDF = document_count / IDF_src
    return log((document_count + 1) / (init_IDF + array([word in words_list and 1 or 0 for word in vocabulary])))


def trained_TFIDF_model_function(document_count
                                 , init_TF
                                 , words_list
                                 , vocabulary
                                 , IDF_src):
    '''
    Examples:
            trained_TFIDF_model_function(document_count=m
                                     , init_TF = init_TF_
                                     , words_list=["隐藏", "端口", "扫描"]
                                     , vocabulary=vocabulary_
                                     , IDF_src=IDF_)
    Args:
        document_count: 历史文档总数
        init_TF : 与训练的单词计数
        words_list: 要续上的文本分词列表
        vocabulary: 词列表
        IDF_src: 原本的IDF模型

    Returns:

    '''
    trained_IDF = trained_IDF_model_function(document_count, words_list, vocabulary, IDF_src)
    trained_TF = trained_TF_model_function(words_list, init_TF, vocabulary)
    # print("TF:\n",trained_TF)
    return trained_TF * trained_IDF


if __name__ == '__main__':
    instructions_text = ['扫描本机所在网段上有哪些主机是存活的',
                         '端口扫描：输入目标主机ip，扫描某台主机开放了哪些端口',
                         '隐藏扫描，输入目标主机ip，只在目标主机上留下很少的日志信息',
                         'UDP端口扫描：输入目标主机ip，扫描目标主机开放了哪些UDP端口',
                         '操作系统识别：输入目标主机ip，查询是哪个系统',
                         '上传或者同步大型项目文件到服务器',
                         '检查本机网段内ip',
                         '查看本机网段内 激活/在线 的设备',
                         '查询本地公网ip', ]
    [
        '上传《网络工具》项目到GPU服务器',
        '上传《网络工具》项目到华为服务器',
        '查询系统运行时间',
        '查询系统开机时间',
        '查询系统历史启动时间',
        '存储盘的位置',
        '酒店开房数据集的位置']

    # print(word_punct_tokenizer)
    TFIDF_ = TFIDF_function(instructions_text)
    IDF_ = TFIDF_["IDF"]
    print("IDF:\n", IDF_)
    init_TF_ = TFIDF_['init_TF']
    TF_ = TFIDF_["TF"]
    m = TFIDF_["document_count"]
    vocabulary_ = TFIDF_["vocabulary_from_TF-IDF"]
    # print(m,vocabulary_,IDF_)

    trained_TFIDF_model_function(document_count=m
                                 , init_TF=init_TF_
                                 , words_list=["隐藏", "端口", "扫描"]
                                 , vocabulary=vocabulary_
                                 , IDF_src=IDF_)
    # print(TFIDF_)
