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
#from numpy import argwhere, zeros, ndarray, array, log, ones
#from data_preprocessing_module import word_punct_tokenizer_for_chinese_function
from tqdm import tqdm
from jieba import cut as jiebacut
from jieba import cut_for_search as jieba_cut_for_search
#from numpy import ndarray, array
from cupy import  zeros, ndarray, array, log, ones , ndarray , array , asnumpy




need_type = list, tuple, ndarray, set

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
    tokens = []
    for iter in jieba_cut_for_search(context_function(character_type_token(original))):
        temp = ''
        number = 0
        iters = (iter.lower() if is_alphabet_function(iter) else iter)
        n = 0
        for char in iters:
            char = str(ord(char))
            n += 1 
            temp+=char
        tokens.append(int(temp))
    return tokens


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


def TFIDF_function_cupy(original_list: (list, ndarray, set, tuple)
                   , filter_stop_words=True):
    '''
    Args:
        word_punct_tokens: 分词后的文章列表
        vocabulary: 词汇表

    Returns: tf 矩阵 横向量为词汇表

    

    '''
    word_punct_tokenizer = data_preprocessing_for_tfidf_function(original_list, filter_stop_words=filter_stop_words)
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
            "vocabulary_from_TF-IDF": vocabulary}  # ,"counter_vector":init_IDF


def TFIWF_function_cupy(original_list: (list, ndarray, set, tuple)
                   , filter_stop_words=True):
    '''
    Args:
        word_punct_tokens: 分词后的文章列表
        vocabulary: 词汇表

    Returns: tf 矩阵 横向量为词汇表

    

    '''
    word_punct_tokenizer = data_preprocessing_for_tfidf_function(original_list, filter_stop_words=filter_stop_words)
    word_punct_tokens, vocabulary = word_punct_tokenizer["word_punct_tokenizer"] , word_punct_tokenizer["vocabulary_set"]
    m, nti = len(word_punct_tokens), len(vocabulary)
    init_TF = zeros((m, nti))
    init_IWF = zeros(nti)
    init_counter_words_for_each_document = zeros(m)
    
    for No, paper_tokens in tqdm(word_punct_tokens.items(),"tfidf训练"):
        vocabulary_of_each_document = len(paper_tokens)
        init_TF += array([paper_tokens.count(word) for word in vocabulary])
        init_IWF += array([word in paper_tokens and 1 or 0 for word in vocabulary])
        init_counter_words_for_each_document[No] = vocabulary_of_each_document
        #print(init_IWF)

    TF = (init_TF.T / init_counter_words_for_each_document).T

    IWF = log(nti / (init_IWF))

    return {"TF-IWF": TF * IWF, "TF": TF, "IDF": IWF, "init_TF": init_TF[0],"document_count": m,"vocabulary_from_TF-IDF": vocabulary ,"counter_vector":init_IWF }
            #, "words_counter": init_IwF.dot(ones(nti)),
              # 
        
