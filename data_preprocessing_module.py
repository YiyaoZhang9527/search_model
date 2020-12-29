#!/usr/bin/env python
"# -*- encoding: utf-8 -*-"
'''
@File  :  data_preprocessing.py
@Author:  manman
@Date  :  2020/11/412:13 上午
@Desc  :
@File  :  
@Time  :  // ::",
@Contact :   408903228@qq.com
@Department   :  my-self
'''
from jieba import cut as jiebacut
from jieba import cut_for_search as jieba_cut_for_search
from numpy import ndarray, array

'''加载标准停用词（标点符号）'''
base_stopwords = ['.', '!', '?', '＂', '＃'
    , '＄', '％', '＆', '＇', '（', '）', '＊'
    , '＋', '，', '－', '／', '：', '；', '＜'
    , '＝', '＞', '＠', '［', '＼', '］', '＾'
    , '＿', '｀', '｛', '｜', '｝', '～', '｟'
    , '｠', '｢', '｣', '､', '\u3000', '、'
    , '〃', '〈', '〉', '《', '》', '「', '」'
    , '『', '』', '【', '】', '〔', '〕', '〖'
    , '〗', '〘', '〙', '〚', '〛', '〜', '〝'
    , '〞', '〟', '〰', '〾', '〿', '–', '—'
    , '‘', '’', '‛', '“', '”', '„', '‟', '…'
    , '‧', '﹏', '﹑', '﹔', '·', '.', '!'
    , '?', '"', '#', '$', '%', '&', "'", '('
    , ')', '*', '+', ',', '-', '/', ':', ';'
    , '<', '=', '>', '@', '[', '\\', ']', '^'
    , '_', '`', '{', '|', '}', '~']


# TODO 中英文字符判断
def is_chinese_function(uchar) -> chr:
    '''
     判断一个unicode是否是汉字
    Args:
        uchar: chart形式的字符
    Returns:

    '''
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


def is_number_function(uchar) -> chr:
    '''
    判断一个unicode是否是数字
    Args:
        uchar:  chart形式的字符

    Returns:

    '''
    if uchar >= u'\u0030' and uchar <= u'\u0039':
        return True
    else:
        return False


def is_alphabet_function(uchar) -> chr:
    '''
    判断一个unicode是否是英文字母
    Args:
        uchar: chart形式的字符
    Returns:

    '''
    """

    """
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
        return True
    else:
        return False


def is_othe_function(uchar) -> chr:
    '''
    判断是否非汉字，数字和英文字符
    Args:
        uchar: chart形式的字符

    Returns:

    '''
    if not (is_chinese_function(uchar) or is_number_function(uchar) or is_alphabet_function(uchar)):
        return True
    else:
        return False


def character_type_token(original) -> str:
    '''

    Args: 字符串形式的文章
        original:

    Returns:

    '''
    '''
    不同字符类型分割
    '''
    make = [0]
    diff = []
    n = 0
    temp = ""
    for char in original:
        if is_chinese_function(char):
            n = 0
        elif is_number_function(char):
            n = 1
        elif is_alphabet_function(char):
            n = 2
        elif is_othe_function(char):
            n = 3
        else:
            n = 4
        make.append(n)
        if (make[-1] - make[-2]) == 0:
            diff.append(char)
        else:
            diff.append("|")
            diff.append(char)
    return "".join(diff).split("|")


# TODO 文章列表预处理函数
def context_function(paper_list) -> (list, set, tuple):
    '''
    连接上下文本列表
    Args: 文章列表
        paper_list:

    Returns:

    '''
    
    return "".join(paper_list)


def tokenize_chinese_function(original) -> str:
    '''
    中文分词
    Args:
        original: 一段文章字符串

    Returns: 分词的列表

    '''
    return [iter.lower() if is_alphabet_function(iter) else iter for iter in
            jieba_cut_for_search(context_function(character_type_token(original)))]


def word_punct_tokenizer_for_chinese_function(article_list: list
                                              , filter_stop_words=False) -> (list, tuple, ndarray, tuple, dict):
    '''

    Args: 对文章列表分词(中文优先)
        article_list: 文章列表
        filter_stop_words: 是否清理词不必要的停用词
        True是过滤基础停用词，Flase是不过滤停用词，
        如果是 list,tuple,dict,set,ndarray等可以
        "in" 判断的结构则过滤定义的停用词

    Returns:
    '''
    m = len(article_list)
    if filter_stop_words == True:
        return {paper_num: filter_stop_words_fumction(tokenize_chinese_function(paper)) for paper, paper_num in
                zip(article_list, range(m))}
    elif filter_stop_words == False:
        return {paper_num: tokenize_chinese_function(paper) for paper, paper_num in zip(article_list, range(m))}
    elif isinstance(filter_stop_words, (list, tuple, dict, ndarray, set)):
        return {
            paper_num: filter_stop_words_fumction(tokenize_chinese_function(paper), stop_words_dict=filter_stop_words)
            for paper, paper_num in
            zip(article_list, range(m))}


def filter_stop_words_fumction(words_list: (list, ndarray)
                               , stop_words_dict=base_stopwords) -> (list, tuple, set):
    '''
    过滤停用词
    Args:
        words_list: 需要过滤的词列表
        stop_words_dict: 停用词表

    Returns: 过滤停用词后的词列表

    '''
    return [word for word in words_list if word not in stop_words_dict]
