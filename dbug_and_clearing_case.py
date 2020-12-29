#!/usr/bin/env python
"# -*- encoding: utf-8 -*-"
'''
@File  :  dbug_and_clearing_case.py
@Author:  manman
@Date  :  2020/11/23:16 下午
@Desc  :
@File  :  
@Time  :  // ::",
@Contact :   408903228@qq.com
@Department   :  my-self
'''

# here put the import lib

import pandas as pd
import sys
from os.path import expanduser
from os import sep
# 加载机器学习模块
import json
import logging
import os
import pandas as pd
from subprocess import getoutput
base_dir = os.path.abspath(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(base_dir)
#from dir_path_config import static_cache_dir




def my_loging_function(obj):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(obj)
    logger.debug(obj)
    logger.warning(obj)
    logger.info(obj)


def display_dir_or_file(args):
    '''
    显示是文件还是文件夹
    Args:
        args:

    Returns:

    '''
    if os.path.isfile(args):
        return 'isfile'
    elif os.path.isdir(args):
        return "isdir"
    else:
        return False


def check_or_create_files_and_folders_function(dirpath):
    '''
    检查文件或文件夹是否存在，如果不存在，则在给出的路径创建文件或者文件夹
    Args:
        dirpath: 文件或文件夹的路径

    Returns:

    '''
    if display_dir_or_file(dirpath) == "isdir":
        return dirpath
    else:
        init_dir_path = os.sep
        for dir_path in dirpath.split(os.sep):
            init_dir_path += "{}{}".format(os.sep, dir_path)
            temp_dir = init_dir_path.replace(os.sep * 2, "")
            isdir_ = os.path.isdir(temp_dir)
            isfile_ = os.path.isfile(temp_dir)
            if isfile_:
                return dirpath
            elif isdir_:
                if len(dirpath) == temp_dir:
                    return dirpath
            else:
                check_leve1 = (isdir_ & isfile_) | (temp_dir != '')
                check_leve2 = isdir_ & (temp_dir != '')
                check_leve3 = check_leve2 != check_leve1

                if "." not in temp_dir:
                    if check_leve3:
                        os.mkdir(temp_dir)
                        print("{0}{2}{1}".format("创建文件夹:\t", "\t成功!", temp_dir))
                else:
                    f = open(temp_dir,'w')
                    f.close()
                    print("{0}{2}{1}".format("创建文件:\t", "\t成功!", temp_dir))
                    return temp_dir


def path_checking_function(args, return_key=False):
    result = dict()
    deeping_level = 0
    level1 = display_dir_or_file(args)
    if level1:
        return {deeping_level: {level1: args}}
    else:
        temp_check_path = os.sep
        for dirname in args.split(os.sep):
            deeping_level += 1
            temp_check_path = os.path.join(temp_check_path, dirname)
            level2 = display_dir_or_file(temp_check_path)
            if return_key:
                if level2 == False:
                    print("需要加入创建文件夹函数")
                    return False
            result.update({deeping_level: {level2: temp_check_path}})

    return result



'''

def convert_dictionary_to_dataframe_function():
    """
    '将字典转换为数据帧'
    """
    pass


def convert_the_dictionary_to_josn_function(data) -> dict:
    """
    '将字典转换为josn'
    """
    if isinstance(data, dict):
        return json.dumps(data, sort_keys=False, indent=4, separators=(',', ': '), ensure_ascii=False)
    else:
        return False


def save_as_csv_format_function(table, table_path=False):
    """
    '保存为csv格式'
    """
    check_path = path_checking_function(table_path)
    if len(check_path) < 2:
        return table.to_csv(table_path)
    dir_checking = list(check_path)[-2]
    file_checking = list(check_path)[-1]
    if ((dir_checking == 'isdir') or (file_checking == "isfile")) and (file_checking.split('.')[-1] == 'csv'):
        if isinstance(table, pd.core.frame.DataFrame):
            return table.to_csv(table_path)
    pass


def save_as_josn_function(data, file_path) -> dict:
    """
    '保存为josn格式'
    """
    if path_checking_function(file_path, return_key=True) == False:
        return "can't find this file or dirs"
    elif os.path.isdir(file_path):
        return False
    josn_data = convert_the_dictionary_to_josn_function(data)
    file = file_path
    init_data = convert_the_dictionary_to_josn_function(data)
    with open(file_path, 'w') as file:
        file.writelines(init_data)
    file.close()


def save_as_text_format_function():
    """
    '保存为text格式'
    """
    pass


def convert_dictionary_to_dataframe_function(data
                                             , save_path=static_cache_dir
                                             , tablename_keyword='convert_dictionary_to_dataframe_function'
                                             , column_names=False):
    """
    '将字典转换为数据帧'
    """
    if isinstance(data, dict):
        checking_level1 = display_dir_or_file(save_path)
        if checking_level1 == "isdir":
            for No, dataMatrix in data.items():
                table_path = "{}{}{}{}{}".format(save_path, tablename_keyword, "_", No, ".csv")
                my_loging_function(table_path)
                init_table = pd.DataFrame(dataMatrix, columns=column_names)
                init_table.to_csv(table_path)


# 将词频字典专为josn文件保存
# save_as_josn_function(feature_dictionary,file_path=os.path.expanduser("~/linux_tools_for_chinese/8_static_cache/word_list.jon"))
# 路径检查函数
# path_checking_function(os.getcwd()+'/test/')
# 将每篇文章的词频保存为DataFrame后转为csv文件保存
# save_as_csv_format_function(word_frequency_of_each_article,table_path=os.path.abspath(os.path.join(os.getcwd()+os.sep,'test.csv')))
# 将文章的onehot字典专为矩阵
# convert_dictionary_to_dataframe_function(onehot_matrix_list,column_names=words_list)
path_checking_function(os.path.expanduser("~/linux_tools_for_chinese/8_static_cache/word_list.jon"), return_key=True)
# 创建文件夹函数
check_or_create_files_and_folders_function(
    os.path.expanduser("~/linux_tools_for_chinese/09_file_manager_component"))
'''