# -*- encoding: utf-8 -*-
'''
@File    :   config.py
@Time    :   2020/12/24 10:57:49
@Author  :   DataMagician 
@Version :   1.0
@Contact :   408903228@qq.com
@Desc    :   None
'''

# here put the import lib

from os import path, getcwd

project_root_directory = path.abspath(getcwd())+"/"#'~/search_model/'
static_cache = path.expanduser("{}{}".format(project_root_directory,"StaticCache/"))
model_data_path = path.expanduser("{}{}".format(static_cache,"model_data/"))
one_hot_encoding_path = "{}{}".format(static_cache,"onehots/")
model_path_json_file = "{}{}".format(static_cache,"model_path.json")
csvfile_path = "{}{}".format(static_cache,"linux_cmd.csv")
linux_cmd_json = "{}{}".format(static_cache,"linux_cmd.json")
natrule_language_column_name = "文字说明"
return_body = "命令"
system_info = "系统"



if __name__ == "__main__":
    print(project_root_directory)