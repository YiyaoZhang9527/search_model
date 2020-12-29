# -*- encoding: utf-8 -*-
'''
@File    :   code_generation.py
@Time    :   2020/10/29 23:14:19
@Author  :   DataMagician 
@Version :   1.0
@Contact :   408903228@qq.com
'''

# here put the import lib

import re
from os import getcwd, walk, mkdir, system
from os import path
from os.path import sep, isdir
import json
import requests
from time import sleep
from tqdm import tqdm


def translate(word):
    url = 'http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule&smartresult=ugc&sessionFrom=null'
    key = {
        'type': "AUTO",
        'i': word,
        "doctype": "json",
        "version": "2.1",
        "keyfrom": "fanyi.web",
        "ue": "UTF-8",
        "action": "FY_BY_CLICKBUTTON",
        "typoResult": "true"
    }
    response = requests.post(url, data=key)
    if response.status_code == 200:
        return response.text
    else:
        return 'Error ,can\'t use it now'


def reuslt(repsonse):
    result = json.loads(repsonse)
    # print("input：%s" % result['translateResult'][0][0]['src'])
    # print("translation：%s" % result['translateResult'][0][0]['tgt'])
    return result['translateResult'][0][0]['tgt']


def translateapi(word):
    return {i: reuslt(translate(i)) for i in tqdm(word, desc="翻译文档开始")}


def translate_main(origin):
    return json.loads(translate(origin))['translateResult'][0][0]['tgt']

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


def spread(arg):
    '''
    广播
    '''
    ret = []
    for i in arg:
        if isinstance(i, (list, tuple, set)):
            ret.extend(i)
        else:
            ret.append(i)
    return ret


def deeping_flatten(mylist):
    '''
    深度平展
    '''
    result = []
    result.extend(
        spread(list(map(lambda x: deeping_flatten(x) if type(x) in (list, tuple, set) else x, mylist))))
    return result


def display_dir_or_file(args):
    '''
    显示是文件还是文件夹
    '''
    if path.isfile(args):
        return 'isfile'
    elif path.isdir(args):
        return "isdir"
    else:
        return False


def check_or_create_files_and_folders_function(dirpath):
    '''

    Args: 检查文件或文件夹是否存在，如果不存在，则在给出的路径创建文件或者文件夹
        dirpath: 文件或文件夹的路径

    Returns:

    '''
    if display_dir_or_file(dirpath) == "isdir":
        return dirpath
    else:
        init_dir_path = sep
        for dir_path in dirpath.split(sep):
            init_dir_path += "{}{}".format(sep, dir_path)
            temp_dir = init_dir_path.replace(sep * 2, "")
            isdir_ = path.isdir(temp_dir)
            isfile_ = path.isfile(temp_dir)
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
                        mkdir(temp_dir)
                        print("{0}{2}{1}".format("创建文件夹:\t", "\t成功!", temp_dir))
                else:
                    f = open(temp_dir, 'w')
                    f.close()
                    print("{0}{2}{1}".format("创建文件:\t", "\t成功!", temp_dir))
                    return temp_dir


def extract_english_function(original):
    '''
    只要英文
    '''
    return re.findall(r'[a-zA-Z]+', original)


def extract_numbers(original):
    '''
    只要数字
    '''
    return re.findall(r'\d+', original)


def extract_numbers_and_english_function(original):
    '''
    只要数字和英文
    '''
    return re.findall(r'[a-zA-Z0-9]', original)


def extract_segment_functions(strings, symbol_to_delete=variable_base):
    """
    '提取其他'
    """
    srcrep = {i: '' for i in symbol_to_delete}
    rep = dict((re.escape(k), v) for k, v in srcrep.items())
    pattern = re.compile("|".join(rep.keys()))
    return pattern.sub(lambda m: rep[re.escape(m.group(0))], strings)


def generation_function(function_names=False
                        , code_snippet=None
                        , dir_path=getcwd() + "/孟哥版"
                        , default_del=variable_function_stop
                        , mode="a+"):
    '''
    批量生成函数
    '''
    check_or_create_files_and_folders_function(dir_path)
    print(dir_path)
    if isinstance(function_names, dict) and isdir(dir_path):

        filename = list(function_names)[0]
        filepath = dir_path + sep + filename + '.py'
        f = open(filepath, mode)
        print(filepath)
        if dir_path and function_names:
            for code_snippet_row in tqdm(code_snippet, desc="代码片段写入"):
                f.write(code_snippet_row)
            f.write("\n")

            for function_items in tqdm(function_names[filename], desc="函数头片段写入"):
                init_function_name, function_API = tuple(*(function_items.items()))
                function_name = "".join(list(
                    map(lambda x: extract_segment_functions(x, symbol_to_delete=default_del) + "_",
                        extract_english_function(init_function_name))))[:-1].lower().replace("_tion_", "")
                # print(function_name)
                function_line = ("def " + function_name + "_function" + "():\n" + '    """\n    ' + repr(
                    function_API) + '\n    """\n' + "    " + "pass\n\n").replace("__", "_")
                # print(function_line)
                f.writelines(function_line)
        f.close()


def generate_variable_names_function(variable_names=False
                                     , dir_path=getcwd() + "/孟哥版"
                                     , default_del=variable_function_stop
                                     , mode="w+"):
    '''
    批量生成变量名
    Args:
        variable_names:
        dir_path:
        default_del:
        mode:

    Returns:

    '''
    if isinstance(variable_names, dict) and isdir(dir_path):

        filename = list(variable_names)[0]
        filepath = dir_path + sep + filename + '.py'
        f = open(filepath, mode)
        if dir_path and variable_names:
            for variable_items in tqdm(variable_names[filename], desc="变量名写入"):
                init_variable_name, annotation = tuple(*(variable_items.items()))
                variable_name = ("".join(list(
                    map(lambda x: extract_segment_functions(x, symbol_to_delete=default_del) + "_",
                        extract_english_function(init_variable_name))))[:-1]).lower()
                variable_line = "variable_" + variable_name + ' = "" #' + annotation + "\n\n"
                f.writelines(variable_line)
        f.close()


def translation_generation(function_names_chinese
                           , code_snippet=["# -*- encoding: utf-8 -*-"]):
    '''

    Args: 中文翻译成函数及注释文档
        function_names_chinese:

    Returns:

    '''
    filename = list(function_names_chinese)[0]
    filter_chinese_from_resturn_data = lambda translate_dict: [{english: chinese} for chinese, english in
                                                               translate_dict.items()]
    translate_filename = "".join([_ + "_" for _ in translateapi([filename])[filename].lower().split()])
    print("正在创建:", translate_filename)
    translate_action = translateapi(deeping_flatten([[chinese for chinese, doc_ in doc_name.items()] for doc_name in
                                                     function_names_chinese[filename]]))
    translate_data = {translate_filename: filter_chinese_from_resturn_data(translate_action)}
    generation_function(translate_data, code_snippet=code_snippet)
    generate_variable_names_function(translate_data, mode="a+")


if __name__ == "__main__":
    function_names = {"test": [{'Saved as CSV': '保存为csv'},
                               {'Save as josn format': '保存为josn格式'},
                               {'Saved in text format': '保存为text格式'},
                               {'Sync files to a remote server': '同步文件到远程服务器'},
                               {'Sync files to the local server': '同步文件到本地服务器'},
                               {'Converts a dictionary josn': '将字典转换为josn'},
                               {'Converts a dictionary data frames': '将字典转换为数据帧'},
                               {'Create folder function': '创建文件夹函数'}]}
    function_names_chinese = {"模版文件": [
        # {'保存为csv': "保存为csv格式"},
        # {'保存为josn格式': "保存为josn格式"},
        # {'保存为text格式': "保存为text格式"},
        # {'同步文件到远程服务器': "同步文件到远程服务器"},
        # {'同步文件到本地服务器': '同步文件到本地服务器'},
        # {"将字典转换为josn": "将字典转换为josn"},
        # {"将字典转换为数据帧": "将字典转换为数据帧"},
        # {"创建文件夹函数": "创建文件夹函数"},
        # {"检查或者创建文件和文件夹": "检查或者创建文件和文件夹"},
        # {"该文档单词出现的次数":"该文档单词出现的次数"},
        # {"单词总数":"单词总数"},
        # {"文档总数":"文档总数"},
        # {"该单词出现的文档数":"该单词出现的文档数"},
        # {"为tfidf做数据预处理":"为tfidf做数据预处理"}
        #{"加载linux命令数据": "DataFrame转文本"},
        #{"加载代码生成配置文件": "加载代码生成配置文件"},
        #{"数据初始化": "数据初始化"},
        #{"数据预处理初始化": "数据预处理初始化"},
        #{"数据预处理模型存储": "数据预处理模型存储"},
        {"概率":"求词概率"},
        {"概率转移矩阵":"概率转移矩阵"},
        {"前向网络":"前向网络"},
        {"后向网络":"后向网络"}
    ]}
    # 读取代码模版

    import_templates = open(
        "/root/孟哥版/StaticCache/temp.py",
        'r').readlines()
    system("figlet start coding")
    #generate_variable_names_function(function_names, mode="w+")
    #generation_function(function_names)
    translation_generation(function_names_chinese , code_snippet=import_templates)
    # search_files_function("8_static_cache", path.expanduser("~/linux_tools_for_chinese/"))
    # print(generates_the_path_function(function_names,new_path="/7_python_functional_script"))
    # print(translateapi(deeping_flatten([[chinese for chinese, doc_ in doc_name.items()] for doc_name in
    #                                    function_names_chinese[list(function_names_chinese)[0]]])))
    system("figlet over!")
