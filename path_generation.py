# -*- encoding: utf-8 -*-
'''
@File    :   path_generation.py
@Time    :   2020/11/04 10:27:37
@Author  :   DataMagician 
@Version :   1.0
@Contact :   408903228@qq.com
@Desc    :   None
'''

# here put the import lib
from os import path, getcwd, walk, sep
from code_generation import extract_segment_functions, extract_english_function, extract_numbers
from tqdm import tqdm
from config import project_root_directory

variable_base = ['！', '？', '｡', '。', '＂', '＃'
    , '＄', '％', '＆', '＇', '（', '）', '＊'
    , '＋', '，', '－', '／', '：', '；', '＜'
    , '＝', '＞', '＠', '［', '＼', '］', '＾'
    , '＿', '｀', '｛', '｜', '｝', '～', '｟'
    , '｠', '｢', '｣', '､', '\u3000', '、', '〃'
    , '〈', '〉', '《', '》', '「', '」', '『'
    , '』', '【', '】', '〔', '〕', '〖', '〗'
    , '〘', '〙', '〚', '〛', '〜', '〝', '〞'
    , '〟', '〰', '〾', '〿', '–', '—', '‘'
    , '’', '‛', '“', '”', '„', '‟', '…', '‧'
    , '﹏', '﹑', '﹔', '·', '.', '!', '?', '"'
    , '#', '$', '%', '&', "'", '(', ')', '*'
    , '+', ',', '-', '/', ':', ';', '<', '='
    , '>', '@', '[', '\\', ']', '^', '_', '`'
    , '{', '|', '}', '~']
variable_function_stop = "func function functions".split() + variable_base

#project_root_directory = '~/search_model/'

def search_files_function(dirpath=path.expanduser(project_root_directory)
                          , target_path=path.expanduser(project_root_directory)
                          , default_del=variable_function_stop):
    '''
    生成文件夹下所有路径的变量模块python文件
    Args:
        target_path: 输出的路径文件目标路径
        dirpath: 搜索的文件夹路径
        default_del: 默认删除的字符

    Returns: {type:[filepath]} 格式化分类的路径目标字典

    '''

    configfile = path.join(target_path, "file_path_config.py")
    print(target_path, configfile)
    configdir = path.join(target_path, "dir_path_config.py")
    f = open(configfile, 'w')
    d = open(configdir, "w")
    file_type_dict = {'csv': [], 'other': [], 'log': [], 'sh': [],
                      'txt': [], 'pyc': [], 'ini': [], 'josn': [],
                      'DS_Store': [], 'py': [], 'json': [], 'ipynb': [],
                      }
    for dir_, folder, files in tqdm(walk(dirpath), desc='路径生成'):
        for file in files:
            temp = path.join(dir_, file)
            split_file_name = temp.split(".")
            file_type = split_file_name[-1]
            clearning_file_name = split_file_name[0]
            if file_type in file_type_dict:
                file_type_dict[file_type].append(temp)
            else:
                file_type_dict["other"].append(temp)
            init_var_name = path.split(temp)[-1].split(".")[0]
            var_name = ("".join(list(map(lambda x: extract_segment_functions(x, symbol_to_delete=default_del) + "_",
                                         extract_english_function(init_var_name))))[:-1]).lower() + "".join(
                extract_numbers(init_var_name))
            if len(var_name):
                code_characters_of_file = "file_" + (var_name) + ' = "' + temp + '"\n'
                code_characters_of_dir = "dir_" + "".join(list(
                    map(lambda x: x + '_', extract_english_function(path.split(temp)[0].split(sep)[-1])))) + ' = "' + \
                                         path.split(temp)[0] + '"\n'

            d.writelines(code_characters_of_dir)
            f.writelines(code_characters_of_file)
    f.close()
    d.close()

    return file_type_dict


if __name__ == '__main__':
    print(search_files_function(dirpath=path.expanduser("~/search_model/")
                          , target_path=path.expanduser("~/search_model/")
                          , default_del=variable_function_stop))  # dirpath, target_path="~/Users/zhangjing/linux_cmd_search_from_NL/"))
