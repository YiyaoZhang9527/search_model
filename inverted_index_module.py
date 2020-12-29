# -*- encoding: utf-8 -*-
'''
@File    :   inverted_index.py
@Time    :   2020/10/06 00:31:42
@Author  :   DataMagician 
@Version :   1.0
@Contact :   408903228@qq.com
'''

# here put the import lib
from data_preprocessing_module import word_punct_tokenizer_for_chinese_function
from numpy import ndarray
from os.path import expanduser


# TODO ：构造到排表类
def inverted_index_function(original: (list, tuple, ndarray, set)
                            , filter_stop_words=True) -> (bool, list, tuple, ndarray, set):
    '''

    Args: 到排表
        original: 文章列表
        filter_stop_word: 是否清理词不必要的停用词
        True是过滤基础停用词，Flase是不过滤停用词，
        如果是 list,tuple,dict,set,ndarray等可以
        "in" 判断的结构则过滤定义的停用词

    Returns: 文章分词字典表示，词典，倒排表
    '''
    if isinstance(filter_stop_words, (bool, list, tuple, ndarray, set)):
        every_paper_token = word_punct_tokenizer_for_chinese_function(original, filter_stop_words=filter_stop_words)
        all_word_tokens = []
        for No, paper_tokens in every_paper_token.items():
            all_word_tokens += paper_tokens
        distinct_words = set(all_word_tokens)

        inverted_index = dict()
        for word in distinct_words:
            for No, paper_tokens in every_paper_token.items():
                if word in paper_tokens:
                    if word not in inverted_index:
                        inverted_index.update({word: [No]})
                    else:
                        inverted_index[word].append(No)
    # elif isinstance(filter_stop_words,(list,tuple,dict,ndarray,set)) :
    #    pass

    return {"article_tokens": every_paper_token, "words_dictionary": distinct_words, "inverted_index": inverted_index}


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
    #path = expanduser("~/linux_tools_for_chinese/4_question_data/question_file.csv")
    #print(inverted_index_function(instructions_text, ["主机", "扫描"]))
