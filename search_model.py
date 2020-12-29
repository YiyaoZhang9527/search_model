#!/usr/bin/env python
"# -*- encoding: utf-8 -*-"
'''
@File  :  search_model.py
@Author:  manman
@Date  :  2020/11/97:52 下午
@Desc  :
@File  :  
@Time  :  // ::",
@Contact :   408903228@qq.com
@Department   :  my-self
'''
import os
import sys
base_dir = os.path.abspath(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(base_dir)
import json
from ast import literal_eval
from jieba import cut_for_search
from numpy import load, zeros, cov, log,array
from pandas import read_csv,DataFrame
from statistics_module import cosangle_function, jaccard_function
from project_init import data_init
from os import path
from statistics_module import sigmoid_function
from project_init import natrule_language_column_name , return_body , system_info



try:
    from dir_path_config import dir_onehots_, dir_model_data_
    from file_path_config import file_tfidf, file_tf \
        , file_probabilistic_feature_of_words_in_each_article \
        , file_document_count_from_tfidf \
        , file_vocabulary_from_tfidf \
        , file_inverted_index, file_probabilistic_feature_vectors \
        , file_probabilistic_feature_vectors \
        , file_probabilistic_feature_dictionary \
        , file_probabilistic_feature_of_words_in_each_article \
        , file_idf \
        , file_init_tf,file_idf
    from project_init import data_init
    from tfidf import trained_TFIDF_model_function

    print("检查配置文件")
except Exception as e:
    data_init.stored_data_preprocessing_model_function()
    from dir_path_config import dir_onehots_, dir_model_data_
    from file_path_config import file_tfidf, file_tf \
        , file_probabilistic_feature_of_words_in_each_article \
        , file_document_count_from_tfidf \
        , file_vocabulary_from_tfidf \
        , file_inverted_index, file_probabilistic_feature_vectors \
        , file_probabilistic_feature_vectors \
        , file_probabilistic_feature_dictionary \
        , file_probabilistic_feature_of_words_in_each_article \
        , file_idf \
        , file_init_tf
    from project_init import data_init
    from tfidf import trained_TFIDF_model_function


def search_inverted_index(keywords, inverted_index_path=file_inverted_index):
    '''加载倒排表'''

    json_inverted_index = open(inverted_index_path, 'r').read()
    load_inverted_index = json.loads(json_inverted_index)
    token_search_text = list(cut_for_search(keywords.lower()))
    return token_search_text, {word: load_inverted_index[word] for word in token_search_text if
                               word in load_inverted_index}


def search_engine_function(keywords
                           , score_functions={None: None}
                           , inverted_index_path=file_inverted_index
                           ):
    '''

    Args:
        keywords:
        score_functions:
        inverted_index_path:

    Returns:

    '''
    # TODO 加载静态缓存数据
    '''分词并加载倒排索引搜索结果'''
    token_search_text, search_for_inverted_index_results = search_inverted_index(keywords, inverted_index_path)

    '''加载缓存的特征字典'''
    json_feature_dictionay = open(file_probabilistic_feature_dictionary, 'r').read()
    feature_dictionay = json.loads(json_feature_dictionay)

    '''加载文本的特征表'''
    feature_table = read_csv(file_probabilistic_feature_of_words_in_each_article)
    feature_matrix = feature_table.to_numpy()

    '''加载文本原文'''
    original_csv = data_init.load_the_linux_command_from_table_function()

    '''搜索内容的字符向量'''
    character_of_search = list(cut_for_search(keywords))

    # TODO <加载tfidf预训练模型缓存>
    tfidf_matrix = load(file_tfidf)
    m_tfidf = load(file_document_count_from_tfidf)
    init_TF_tfidf = load(file_init_tf)
    vocabulary_tfidf = literal_eval(open(file_vocabulary_from_tfidf, "r").readlines()[-1])
    IDF_src_tfidf = load(file_idf)#path.expanduser("~/Documents/算法笔记/000_数据分析demo/拉勾网数据分析/linux_cmd_search_from_NL/linux_cmd_search_from_NL/StaticCache/model_data/IDF.npy"))

    token_search_text_tfidf = trained_TFIDF_model_function(
        document_count=m_tfidf
        , init_TF=init_TF_tfidf
        , words_list=token_search_text
        , vocabulary=vocabulary_tfidf
        , IDF_src=IDF_src_tfidf
    )

    # TODO 提取输入搜索文本关键词的特征向量
    search_word_dictionay = zeros(feature_matrix.shape[-1])
    for search_word in token_search_text:
        n = -1
        for word in feature_dictionay:
            n += 1
            if word == search_word:
                feature = feature_dictionay[word]
                search_word_dictionay[n] = feature
        n = 0

    # TODO 整理除要搜索的文章
    index_of_articles_to_be_searched = dict()
    for word, inverted_index in search_for_inverted_index_results.items():
        for No in inverted_index:
            if No in index_of_articles_to_be_searched:
                index_of_articles_to_be_searched[No]["counter"] += 1
                index_of_articles_to_be_searched[No]["words"].append(word)
            else:
                index_of_articles_to_be_searched.update({No: {"counter": 1, "words": [word]}})

    '''倒序排序到排表搜索结果'''
    sorted_index_of_articles_to_be_searched = sorted(index_of_articles_to_be_searched.items(),
                                                     key=lambda x: x[1]["counter"], reverse=True)

    # TODO  搜索开始
    m = len(sorted_index_of_articles_to_be_searched)
    temp1 = dict()
    for iter in sorted_index_of_articles_to_be_searched:
        # TODO 提取搜索及显示需要的数据结构
        '''提取原文行号'''
        No = iter[0]
        '''提取排序后的到排表信息'''
        info = iter[-1]
        '''提取本行文本中符合条件的单词有多少'''
        counter = info["counter"]
        '''提取本行文本的特征向量'''
        original_eigenvector = feature_matrix[No]
        '''提取本行文本中符合搜索文本的单词表'''
        words = info["words"]
        '''提取原文csv中的行内容'''
        original = original_csv[No:No + 1].to_dict()
        instructions = original[natrule_language_column_name][No]
        cmd = original[return_body][No]
        system = original[system_info][No]
        tfidf_vector = tfidf_matrix[No]
        '''分割原文字符'''
        tokenize_instructions = list(cut_for_search(instructions))

        # TODO 使用tfidf
        cosangle_tfidf = cosangle_function(tfidf_vector, token_search_text_tfidf)['cosangle']

        # TODO 分值计算模块
        '''余弦相似度'''
        cosangle_value = cosangle_function(original_eigenvector, search_word_dictionay)['cosangle']

        '''亚卡尔系数'''
        jaccard_value = sigmoid_function(jaccard_function(tokenize_instructions, character_of_search))
        '''字符串的亚卡尔系数'''
        jaccard_value_of_character = sigmoid_function(jaccard_function(list(keywords), list(instructions)))
        '''词向量的协方差'''
        cov_of_word = sigmoid_function(cov(original_eigenvector, search_word_dictionay)[0, 1])

        score = cosangle_tfidf
        test_score = sum([cosangle_tfidf, cosangle_value, jaccard_value, jaccard_value_of_character, cov_of_word])

        temp = {No: {"分值": score, "重叠词计数": counter, "TFIDF的余弦值": cosangle_tfidf
            , "文本长度": m, "词向量的余弦值": cosangle_value
            , '词向量的协方差': sigmoid_function(cov_of_word), "字符串的亚卡尔系数": jaccard_value_of_character
            , "单词计算的亚卡尔系数": jaccard_value, "本文重叠词列表": words
            , "文本说明": instructions, "系统命令": cmd, "系统": system, "测试分": test_score}}
        temp1.update(temp)

    #return sorted(result.items(), key=lambda x: x[1]['分值'], reverse=True)
    No = -1
    result = []
    for index_, items_ in sorted(temp1.items(), key=lambda x: x[1]['分值'], reverse=True):
        No += 1
        result.append(array([iter[-1] for iter in items_.items()],dtype=object))
    column_names = ['分值' , '重叠词计数' , 'TFIDF的余弦值' ,'文本长度'
        , '词向量的余弦值' , '词向量的协方差' , '字符串的亚卡尔系数'
        ,'单词计算的亚卡尔系数' , '本文重叠词列表' , '文本说明'
        , '系统命令' , '系统' , '测试分']
    df = array(result)
    df = DataFrame(df,columns=column_names)
    result_df = df.sort_values(by="重叠词计数", ascending=False)
    return result_df


if __name__ == '__main__':
    
    print("请输入搜索关键词:")
    keywords = input()
    start = search_engine_function(keywords)
    No = 0
    for line in start.T.to_dict().items():
        No += 1
        print("\n")
        for body in line[-1].items():
            print(body)
        print("{}{}{}".format("当前是第",No.__repr__(),"个搜索结果"))
    '''
    result = []
    for index_, items_ in search_engine_function(keywords):
        No += 1
        result.append(array([iter[-1] for iter in items_.items()]))
    column_names = ['分值' , '重叠词计数' , 'TFIDF的余弦值' ,'文本长度'
        , '词向量的余弦值' , '词向量的协方差' , '字符串的亚卡尔系数'
        ,'单词计算的亚卡尔系数' , '本文重叠词列表' , '文本说明'
        , '系统命令' , '系统' , '测试分']
    df = array(result)
    df = DataFrame(df,columns=column_names)
    result_df = df.sort_values(by="重叠词计数", ascending=False)
    print(result_df)
    '''


    # print('-',iter)

