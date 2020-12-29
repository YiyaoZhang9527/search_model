#!/usr/bin/env python
"# -*- encoding: utf-8 -*-"
'''
@File  :  project_init.py
@Author:  manman
@Date  :  2020/11/410:19 下午
@Desc  :
@File  :  
@Time  :  // ::",
@Contact :   408903228@qq.com
@Department   :  my-self
'''
import os
import sys
base_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.split(base_dir)[0])
sys.path.append(base_dir)
from character_encoding import character_encoding_function
from tfidf import TFIDF_function
from inverted_index_module import inverted_index_function
from os import path, getcwd
from pandas import DataFrame, read_csv, read_json
from numpy import arange, zeros, array
from datetime import datetime, timezone, timedelta
from numpy import save as npsave
from numpy import savez as npsavez
from numpy import savetxt
import json
from path_generation import search_files_function
from tqdm import tqdm
from config import project_root_directory,static_cache,model_data_path,one_hot_encoding_path,model_path_json_file,csvfile_path,linux_cmd_json,natrule_language_column_name,return_body,system_info

china_tz = timezone(timedelta(hours=8))
china_tz = timezone(timedelta(hours=8), 'Asia/Shanghai')
dt = datetime.now(china_tz)
dt = datetime.now(china_tz)
print(china_tz.tzname(dt))


#project_root_directory = '~/search_model/'

#print(read_csv(csvfile_path))


class data_init:

    def __init__(self, object, init_key=False):
        '''
        数据预处理初始化
        Args:
            data:
        '''
        self.data = object
        if init_key == True:
            data_init.data_initialization_function()
        else:
            pass

        # super(path,self).__init__()

    @staticmethod
    def create_the_linux_command_function(
            table_path=None
            , column_names=["documentation", "command", "system"
                , "use_frequency", "counter_user"
                , "keywords_of_user_search", "date", "score"
                , "key_point_model_sed"]):
        '''

        Args:
            table_path:
            column_names:

        Returns:
linux_cmd.csv
        '''
        define_table_path = csvfile_path
        if table_path == None:
            table_path = define_table_path
        empty = zeros((len(column_names), len(column_names)))
        table = DataFrame(empty, columns=column_names)
        table.to_csv(table_path)

    @staticmethod
    def load_the_linux_command_from_table_function(table_path=None):

        '''
        加载linux的csv文件成为DataFrame
        Args:
            path:

        Returns:

        '''

        define_table_path = csvfile_path
        define_table_json = linux_cmd_json
        if table_path == None:
            table_path = define_table_path
        if path.isfile(table_path):
            table = read_csv(table_path)
            newtable = table.drop(columns=["Unnamed: 0"])
            #print(newtable)

            '''
            columns_for_json = newtable.columns
            documentation, command \
                , system, use_frequency \
                , counter_user, keywords_of_user_search \
                , date, score, key_point_model_sed = \
                newtable["documentation"], newtable['command'] \
                    , newtable["system"], newtable['use_frequency'] \
                    , newtable['counter_user'], newtable['keywords_of_user_search'] \
                    , newtable['date'], newtable['score'] \
                    , newtable['key_point_model_sed']
            '''

            table_to_json = newtable.T.to_json()
            jf = open(define_table_json, mode="w+", encoding="'utf-8'")
            jf.write(table_to_json)
            jf.close()


        else:
            data_init.create_the_linux_command_function()
        return newtable

    @staticmethod
    def ndarrays_to_dataframe_function(data
                                       , filename
                                       , column_names
                                       , filepath=one_hot_encoding_path):
        '''
        批量保存矩阵成为csv表格
        Args:
            data:
            path:
            filename:
            column_names:

        Returns:

        '''
        for i in range(len(data)):
            table = DataFrame(data[i], columns=column_names)
            table.to_csv(filepath + filename + i.__repr__() + '.csv')

    @staticmethod
    def save_set_function(data
                          , filename
                          , filepath=static_cache
                          ):
        f = open(filepath + filename, 'w+')
        for word in data:
            f.writelines(str(data) + "\n")
        f.close()

    @staticmethod
    def save_json_function(data
                           , filename
                           , filepath=static_cache):

        f = open(filepath + filename + '.json', 'w+', encoding='UTF-8')
        print(f)
        f.writelines(json.dumps(data))
        f.close()

    @staticmethod
    def initialization_data_preprocessing_function(object):
        '''

        Args: '数据预处理初始化'
            object:
            *path:

        Returns:

        '''

        variable_character_encoding = character_encoding_function(object)
        INIT_TFIDF = TFIDF_function(object)
        variable_TFIDF = INIT_TFIDF['TF-IDF']
        variable_TF = INIT_TFIDF["TF"]
        variable_IDF = INIT_TFIDF["IDF"]
        variable_words_counter_from_TFIDF = INIT_TFIDF["words_counter"]
        variable_document_count = INIT_TFIDF["document_count"]
        variable_init_TF = INIT_TFIDF["init_TF"]
        variable_vocabulary_from_TFIDF = INIT_TFIDF["vocabulary_from_TF-IDF"]

        INIT_inverted_index = inverted_index_function(object)
        variable_article_tokens = INIT_inverted_index["article_tokens"]
        variable_words_dictionary = INIT_inverted_index["words_dictionary"]
        variable_inverted_index = INIT_inverted_index["inverted_index"]
        variable_onehots_encoding_for_each_article = variable_character_encoding["onehots_encoding_for_each_article"]
        variable_counter_vectors = variable_character_encoding["counter_vectors"]
        variable_probabilistic_feature_of_words_in_each_article = variable_character_encoding[
            "probabilistic_feature_of_words_in_each_article "]
        variable_vocabulary = variable_character_encoding["vocabulary"]
        variable_probabilistic_feature_dictionary = variable_character_encoding["probabilistic_feature_dictionary"]
        variable_probabilistic_feature_vectors = variable_character_encoding["probabilistic_feature_vectors"]
        variable_counter = variable_character_encoding["counter"]

        return {"TFIDF": {
            "data": variable_TFIDF
            , "save": lambda data, path: npsave(path + "TFIDF", data)
            , "state": True}
            , "TF": {
                "data": variable_TF
                , "save": lambda data, path: npsave(path + "TF", data)
                , "state": True}
            , "IDF": {
                "data": variable_IDF
                , "save": lambda data, path,: npsave(path + "IDF", data)
                , "state": True}

            , "init_TF": {
                "data": variable_init_TF
                , "save": lambda data, path,: npsave(path + "init_TF", data)
                , "state": True}
            , "words_counter_from_TFIDF": {
                "data": variable_words_counter_from_TFIDF
                , "save": lambda data, path,: npsave(path + "words_counter_from_TFIDF", data)
                , "state": True}
            , "document_count": {
                "data": variable_document_count
                , "save": lambda data, path,: npsave(path + "document_count_from_TFIDF", data)
                , "state": True}

            , "vocabulary_from_TFIDF": {
                "data": variable_vocabulary_from_TFIDF
                , "save": lambda data, path,: data_init.save_set_function(data, "vocabulary_from_TFIDF", path)
                , "state": True
            }

            , "inverted_index": {
                "data": variable_inverted_index
                , "save": lambda data, path: data_init.save_json_function(data, 'inverted_index', path)
                , "state": False}  # 到排表
            , "article_tokens": {
                "data": variable_article_tokens
                , "save": lambda data, path: data_init.save_json_function(data, "article_tokens", path)
                , "state": False}
            , "words_dictionary": {
                "data": variable_words_dictionary
                , "save": lambda data, path: data_init.save_set_function(data, 'words_dictionary',
                                                                         path)
                , "state": False}
            , "onehots_encoding_for_each_article": {
                "data": variable_onehots_encoding_for_each_article
                , "save": lambda data, path: data_init.ndarrays_to_dataframe_function(data
                                                                                      ,
                                                                                      filename="onehots_encoding_for_each_article"
                                                                                      ,
                                                                                      column_names=variable_vocabulary)
                , "state": False}  # 每一篇文章的onehot编码
            , "counter_vectors": {
                "data": variable_counter_vectors
                , "save": lambda data, path: data.to_csv(path + "counter_vectors.csv")
                , "state": True}  # 计数向量
            , "probabilistic_feature_of_words_in_each_article": {
                "data": variable_probabilistic_feature_of_words_in_each_article
                , "save": lambda data, path: data.to_csv(path + "probabilistic_feature_of_words_in_each_article.csv")
                , "state": True}
            # 每篇文章的概率特征
            , "vocabulary": {
                "data": variable_vocabulary
                , "save": lambda data, path: data_init.save_set_function(data, "vocabulary", path)
                , "state": False}  # 词汇表
            , "probabilistic_feature_dictionary": {
                "data": variable_probabilistic_feature_dictionary
                ,
                "save": lambda data, path: data_init.save_json_function(data, 'probabilistic_feature_dictionary', path)
                , "state": False}  # 概率特征字典
            , "probabilistic_feature_vectors": {
                "data": variable_probabilistic_feature_vectors
                , "save": lambda data, path,: npsave(path + "probabilistic_feature_vectors", data)
                , "state": True}  # 概率特征向量
        }  # 总单词数量计数

    '''
    IDF_ = TFIDF_["IDF"]
    init_TF_ = TFIDF_['init_TF']
    m = TFIDF_["document_count"]
    vocabulary_ = TFIDF_["vocabulary_from_TF-IDF"]
    
    '''

    @staticmethod
    def stored_data_preprocessing_model_function(object=None):
        '''

        Args:
            object: 需要预处理的文本list

        Returns:

        '''
        """
        '数据预处理模型存储'
        """
        if object == None:
            object = data_init.load_the_linux_command_from_table_function()[natrule_language_column_name].to_list()
        define_model_dir = model_data_path
        '''初始化模型'''
        init_metadatas = data_init.initialization_data_preprocessing_function(object)
        '''保存所有预处理模型'''
        file_path_ini = dict()
        for model in tqdm(init_metadatas, desc="存储预处理模型"):
            metadata = init_metadatas[model]
            data = metadata['data']
            save_func = metadata['save']
            state = metadata["state"]
            if state:
                save_func(data, define_model_dir)
            else:
                try:
                    save_func(data, define_model_dir)
                except Exception as e:
                    print(model, data)
        '''保存模型的存储路径'''
        jf = open(model_path_json_file, 'a')
        dict_to_json = json.dumps(file_path_ini)
        jf.writelines(dict_to_json)
        '''生成模型的路径变量'''
        search_files_function()
        return file_path_ini


def __del__(self):
    pass





if __name__ == '__main__':
    #print(data_init.create_the_linux_command_function())
    print(data_init.load_the_linux_command_from_table_function()[natrule_language_column_name].to_list())
    instructions_text = data_init.load_the_linux_command_from_table_function(csvfile_path)[natrule_language_column_name].to_list()
    '''
        ['扫描本机所在网段上有哪些主机是存活的',
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
                         '''
    # print(data_init.initialization_data_preprocessing_function(instructions_text))
    print(data_init.stored_data_preprocessing_model_function(instructions_text))
