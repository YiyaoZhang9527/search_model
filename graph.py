#!/usr/bin/env python
"# -*- encoding: utf-8 -*-"
'''
@File  :  graph.py
@Author:  manman
@Date  :  2020/12/2811:03 下午
@Desc  :
@File  :  
@Time  :  // ::",
@Contact :   408903228@qq.com
@Department   :  my-self
'''
# !/usr/bin/env python
# coding: utf-8

# In[1]:


import jieba
# import cupy as cp
import seaborn as sns
import numpy as np
sns.set()

original = """
推荐热点视频图片娱乐科技汽车体育财经军事国际时尚旅游更多
发文
张慢慢的走读世界反馈侵权投诉头条产品
首页 / 正文

搜索
5
转发
微博
Qzone
微信
一文带你全面了解图神经网络
原创华章科技2020-01-07 21:10:00
导读：2020年一开年，就有神秘大咖跟数据叔说：2020，两个事必火，一个是中台，一个是图神经网络。大咖还说，有数据为证，因为2019-2020年之间，图神经网络成为各大顶会的增长热词。
本文就带你全面了解风口上的图神经网络。
​作者：刘忠雨

来源：华章科技

一文带你全面了解图神经网络
2019年初，短短一个月内有三篇关于图神经网络的综述论文接连发表，这预示着2019年学术界对图神经网络的关注将显著提高，事实也确实如此。

2019 年包括深度学习、计算机视觉、文本处理以及数据挖掘在内的顶级会议，图学习相关的论文较之于前都有十分明显的增长。而就 AAAI 2020 的情况来看，这个趋势还在不断扩大。Graph Neural Network 在2019年到 2020年之间力压 deep learning、gan等，成为各大顶会的增长热词。

一文带你全面了解图神经网络
除了学术界，工业界的动作也非常迅速，2019年初阿里开源工业级图深度学习框架 Euler，并在 KDD、CIKM 等会议发表多篇关于图神经网络在推荐、风控等业务中的应用论文。

旷世、腾讯在内的很多知名企业已经将图神经网络应用于视觉、推荐、风控等业务当中。2019 年，阿里达摩院发布未来十大科技趋势认为超大规模图神经网络系统将赋予机器常识。

2019年可以说是图神经网络元年。

01 什么是图神经网络？
1. 图和属性图

要了解图神经网络，首先要了解图。图是由节点和边组成的，如下图所示。一般图中的节点表示实体对象（比如一个用户、一件商品、一辆车、一张银行卡等都可以作为节点），边代表事件或者实体之间的特殊关系（比如用户和商品之间的购买关系）。

一文带你全面了解图神经网络
在数学中，我们一般使用邻接矩阵来表示图，如上图右边所示。邻接矩阵中的值为 1 表示节点之间有边，即有连接关系。所以邻接矩阵其实很好的将图的这种结构信息表达出来了。

还要介绍一个概念是属性图。就是说，图中的节点和边都带有属性（这是一种信息）。如下图所示：

一文带你全面了解图神经网络
这个图里的用户节点有姓名、性别，话题节点具体的话题类别，公司节点有名称，注册时间等属性信息。边也可以有属性信息，比如开始工作时间是边“工作于”的一种属性。所以，属性图就是节点和边带有自己的属性信息，同时每个节点又有自己的拓扑结构信息。这是工业界最常用的一种图表示方法，因为我们需要更丰富的信息。

前几年神经网络很火，相信大家对神经网络都有一定的了解。图神经网络就是将图数据和神经网络进行结合，在图数据上面进行端对端的计算。

2. 图神经网络的计算机制

单层的神经网络计算过程：

一文带你全面了解图神经网络
相比较于神经网络最基本的网络结构全连接层（MLP），特征矩阵乘以权重矩阵，图神经网络多了一个邻接矩阵。计算形式很简单，三个矩阵相乘再加上一个非线性变换。

一文带你全面了解图神经网络
一文带你全面了解图神经网络
图神经网络的计算过程总结起来就是聚合邻居。如下面的动图所示，每个节点都在接收邻居的信息。为了更加全面的刻画每个节点，除了节点自身的属性信息，还需要更加全面的结构信息。所以要聚合邻居，邻居的邻居.....

一文带你全面了解图神经网络
图神经网络是直接在图上进行计算，整个计算的过程，沿着图的结构进行，这样处理的好处是能够很好的保留图的结构信息。而能够对结构信息进行学习，正是图神经网络的能力所在，下面我们就来看看图神经网络为什么强大？

02 图神经网络的强大能力
现实生活中的大量的业务数据都可以用图来表示。万事万物皆有联系，节点+关系这样一种表示足以包罗万象。

比如人类的社交网络，个体作为节点，人与人之间的各种关系作为边；电商业务中，用户和商品也可以构建成图网络；而物联网、电网、生物分子这些是天然的节点+关系结构；甚至，可以将实物物体抽象成 3D 点云，以图数据的形式来表示。图数据可以说是一种最契合业务的数据表达形式。

一文带你全面了解图神经网络
图神经网络的强大能力我认为可以归纳为三点：

对图数据进行端对端学习
擅长推理
可解释性强
1. 端对端学习

近几年，深度学习带来了人脸识别、语音助手以及机器翻译的成功应用。这三类场景的背后分别代表了三类数据：图像、语音和文本。

深度学习在这三类场景中取得突破的关键是它背后的端对端学习机制。端对端代表着高效，能够有效减少中间环节信息的不对称，一旦在终端发现问题，整个系统每一个环节都可以进行联动调节。

既然端对端学习在图像、语音以及文本数据上的学习是如此有效，那么将该学习机制推广到具有更广泛业务场景的图数据就是自然而然的想法了。

这里我们引用 DeepMind 论文中的一段话，来说明其重要性：

我们认为，如果 AI 要实现人类一样的能力，必须将组合泛化（combinatorial generalization）作为重中之重，而结构化的表示和计算是实现这一目标的关键。正如生物学里先天因素和后天因素是共同发挥作用的，我们认为“人工构造”（hand-engineering）和“端到端”学习也不是只能从中选择其一，我们主张结合两者的优点，从它们的互补优势中受益。
2. 擅长推理

业界认为大规模图神经网络是认知智能计算强有力的推理方法。图神经网络将深度神经网络从处理传统非结构化数据（如图像、语音和文本序列）推广到更高层次的结构化数据（如图结构）。
大规模的图数据可以表达丰富和蕴含逻辑关系的人类常识和专家规则，图节点定义了可理解的符号化知识，不规则图拓扑结构表达了图节点之间的依赖、从属、逻辑规则等推理关系。
以保险和金融风险评估为例，一个完备的 AI 系统不仅需要基于个人的履历、行为习惯、健康程度等进行分析处理，还需要通过其亲友、同事、同学之间的来往数据和相互评价进一步进行信用评估和推断。基于图结构的学习系统能够利用用户之间、用户与产品之间的交互，做出非常准确的因果和关联推理。
——达摩院2020十大科技趋势白皮书
3. 可解释性强

图具有很强的语义可视化能力，这种优势被所有的 GNN 模型所共享。比如在异常交易账户识别的场景中，GNN 在将某个账户判断为异常账户之后，可以将该账户的局部子图可视化出来，如下图所示：

一文带你全面了解图神经网络
我们可以直观地从子图结构中发现一些异常模式，比如同一设备上有多个账户登录，或者同一账户在多个设备上有行为。还可以从特征的维度，比如该账户与其他有关联的账户行为模式非常相似（包括活跃时间集中，或者呈现周期性等），从而对模型的判断进行解释。

论文 “GNNExplainer: Generating Explanations for Graph Neural Networks” 提供了一种自动从子图中提取重要子图结构和节点特征的方法，可以为 GNN 的判断结果提供重要依据。

03 图神经网络的应用
图数据无处不在，图神经网络的应用场景自然非常多样。笔者在这里选择一部分应用场景为大家做简要的介绍，更多的还是期待我们共同发现和探索。

一文带你全面了解图神经网络
1. 计算机视觉

在计算机视觉的应用有根据提供的语义生成图像，如下图所示（引用）。输入是一张语义图，GNN通过对“man behind boy on patio”和“man right of man throwing firsbee”两个语义的理解，生成了输出的图像。

一文带你全面了解图神经网络
▲图片来源：https://arxiv.org/pdf/1804.01622.pdf

再说说视觉推理，人类对视觉信息的处理过程往往参杂着推理。比如下图的场景中，左上角第4个窗户虽然有部分遮挡，我们仍可以通过其他三扇窗户推断出它是窗户；再看右下角的校车，虽然车身不完整，但我们可以通过这个车身颜色推断出其是校车。

一文带你全面了解图神经网络
▲图片来源：https://arxiv.org/pdf/1803.11189.pdf

人类可以从空间或者语义的维度进行推理，而图可以很好的刻画空间和语义信息，让计算机可以学着像人类一样，利用这些信息进行推理。

一文带你全面了解图神经网络
▲图片来源：https://arxiv.org/abs/1803.11189

当然还有动作识别，视觉问答等应用，这里我们就不一一列举了，感兴趣的同学推荐大家阅读文章：

图像生成
https://arxiv.org/pdf/1804.01622.pdf
视觉推理
https://arxiv.org/pdf/1803.11189.pdf
2. 自然语言处理

GNNs 在自然语言处理中的应用也很多，包括多跳阅读、实体识别、关系抽取以及文本分类等。多跳阅读是指给机器有很多语料，让机器进行多链条推理的开放式阅读理解，然后回答一个比较复杂的问题。在2019年，自然语言处理相关的顶会论文使用 GNN 作为推理模块已经是标配了。

多跳阅读：
https://arxiv.org/pdf/1905.06933.pdf
关系抽取和文本分类应用也十分多，这里推荐大家阅读：

https://mp.weixin.qq.com/s/i2pgW4_NLCB1Bs3qRWRYoA
3. 生物医疗

我们在高中都接触过生物化学，知道化合物是由原子和化学键构成的，它们天然就是一种图数据的形式，所以图神经网络在生物医疗领域应用特别广泛。包括新药物的发现、化合物筛选、蛋白质相互作用点检测、以及疾病预测。

据笔者所知，目前国外包括耶鲁、哈佛，国内像北大清华都有很多实验室研究图神经网络在医学方面的应用，而且我相信这会是图神经网络最有价值的应用方向之一。

除了上述的方向，还有像在自动驾驶和 VR 领域会使用的 3D 点云；与近两年同样很火的知识图谱相结合；智慧城市中的交通流量预测；芯片设计中的电路特性预测；甚至还可以利用图神经网络编写代码。

目前在真正在工业场景中付诸应用，并取得了显著成效的场景主要有两个，一是推荐，二是风控。

4. 工业应用之推荐

推荐是机器学习在互联网中的重要应用。互联网业务中，推荐的场景特别说，比如内容推荐、电商推荐、广告推荐等等。这里，我们介绍三种图神经网络赋能推荐的方法。

（1）可解释性推荐

可解释性推荐，就是不仅要预测推荐的商品，还要给出推荐的理由。推荐中有一个概念叫元路径。在电影推荐的场景里，如下图所示。我们用 U 表示用户，用 M 表示电影，那么 UUM 是一条元路径。它表示一位用户关注了另一位用户，那么我们可以将用户看过的电影，推荐给关注他的人。

当然，还有比如 UMUM 表示与你看过相同电影的人还在看什么电影这条路径；UMTM 表示与你看过同一类型电影的路径.....元路径有很多，不同元路径对于不同的业务语义。在这个场景中，图神经网络模型有两个任务，一个是推荐影片给用户，二是给出哪条元路径的权重更高。而这正式 GNN 可解释性的体现。

一文带你全面了解图神经网络
▲论文链接：http://www.shichuan.org/doc/47.pdf

（2）基于社交网络的推荐

利用用户之间的关注关系，我们也可以实现推荐。用户的购买行为首先会受到其在线社交圈中朋友的影响。如果用户 A 的朋友是体育迷，经常发布关于体育赛事、体育明星等信息，用户 A 很可能也会去了解相关体育主题的资讯。

其次，社交网络对用户兴趣的影响并非是固定或恒定的，而是根据用户处境（Context）动态变化的。举例来说，用户在听音乐时更会受到平时爱好音乐的朋友影响，在购买电子产品时更会受到电子发烧友的朋友影响。目前有许多的电商平台，包括像京东、蘑菇街、小红书等都在尝试做基于社交的推荐。

一文带你全面了解图神经网络
▲论文链接：http://www.cs.toronto.edu/~lcharlin/papers/fp4571-songA.pdf

（3）基于知识图谱的推荐

要推荐的商品、内容或者产品，依据既有的属性或者业务经验，可以得到他们之间很多的关联信息，这些关联信息即是我们通常说的知识图谱。知识图谱可以非常自然地融合进已有的用户-商品网络构成一张更大、且包含更加丰富信息的图。

一文带你全面了解图神经网络
▲论文链接：https://arxiv.org/pdf/1803.03467.pdf

其实不管是社交网络推荐，还是知识图谱，都是拿额外的信息补充到图网络中。既能有聚合关系网络中复杂的结构信息，又能囊括丰富的属性信息，这就是图神经网络强大的地方。

国外图片社交媒体 Pinterest 发表了利用图神经网络做推荐的模型 PinSage 。大家应该也都比较熟悉了，这里就不再赘述了。

5. 工业应用之风控

我们公司利用图来做风控还是有一些时间了。我们的业务场景中每天都会有很多网络请求，一个请求过来，需要实时的判断这是真实用户还是机器流量。一个简单的模型，使用的数据包括设备ID、IP、用户以及他们的行为数据，构图如下：

一文带你全面了解图神经网络
我们全网一天有将近十亿次网络请求，全部日志构成一张图，包括 1.6 亿节点、12亿边。相较于之前的深度学习方法，AUC 指标提升1.2倍，上线测试该模型的稳定性指标最优，提升 1.5 倍。

去年 12 月我们做了一个项目，总共是 2800 万的网站业务（包括 IP、UA、域名、访问时间等等）以及第三方的威胁情报库数据。该场景下的任务是预测网络请求是否为恶意请求。比如说黑产可能通过 POST 端注入一些恶意代码，操作数据库。

我们的解决方案很简单，只使用了 4 个字段。某个请求在某个IP哪个域名注入了某种脚本，以及 POST 特征码。前三个字段（请求事件ID，IP，域名）构成了一个图，是模型的输入。最后一个 POST 特征码是网络需要预测的或者说在预测时候的监督信号。

我们模型结果的输出首先是攻击语言的识别，2800 万的流量里面有 70% 的是异常流量，分别来自于六个不同的攻击语言，并且识别出相应的作案手段。然后也发现第三方威胁 IP 库，实际上是有大量的误封的。攻击目标的识别，输出了对应被攻击的域名列表 2000 条。仅仅用了4个字段，就完成了异常流量、攻击目的以及攻击语言的识别。

一文带你全面了解图神经网络
在很多互联网营销场景里存在着大量薅羊毛的恶意账户。识别恶意账户就是对图中的用户节点进行分类。薅羊毛用户有一个非常本质的特点，就是他们的行为模式非常相似。由于他们的资源不是无限的，会共用一些设备账号，包括手机号等。所以他们的数据有非常多的关联。

另一个特点是在短时间内活跃，薅羊毛用户往往是在做活动的时间段行为非常活跃，而在其他的业务场景里面，活跃度很低，具有短时高频的特点。所以，要识别这样的恶意账户，我们主要用了两份信息，第一个是资源的关联信息，另一个就是时间上的行为信息， 他们和正常用户在时间上的行为模式是不太一样的。

GNN 可以端对端的去学习这两类信息。这个场景阿里也发了一篇论文去讲，恶意账户的识别，最后相比较其他的方法，比如像图分区去挖掘这种团伙，包括像 GBDT 这种浅层的机器学习模型，效果是比较突出的。

论文链接：
http://shichuan.org/hin/topic/Others/2018.CIKM%202018%20Heterogeneous%20Graph%20Neural%20Networks%20for%20Malicious%20Account%20Detection.pdf
图数据包罗万象，图神经网络的应用场景将会非常丰富。从 2020 年 AAAI 和 ICLR 的情况来看，图神经网络在学术界已经掀起了一阵新的潮流。当然，工业界也会迎来更多的投入和关注，毕竟图数据是最贴合业务的数据。

这里要向大家推荐一本关于图神经网络的书《深入浅出图神经网络》。这是我以及我的公司极验图数据团队结合自己在图神经网络领域的研究和实践经验撰写的一本入门书籍，从原理、算法、实现、应用 4 个维度为大家详细全面的讲解了图神经网络。希望能够对大家学习和利用图神经网络技术有所帮助。

一文带你全面了解图神经网络

"""

# In[2]:


# !/usr/bin/env python
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
from numpy import ndarray

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
    , '_', '`', '{', '|', '}', '~', '\n', '。'
    , ' ', '.....']


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


word_list = filter_stop_words_fumction(jieba.cut(original))


def graph_DAG_function(data, skip_n=1):
    '''

    Args: 有向无环图
        data: 序列数据
        skip_n: 跳的步数

    Returns: 有向无环图

    '''
    graph = {}
    for i in range(len(data) - skip_n):
        node = data[i]
        first, end = i + 1, i + skip_n + 1
        if node not in graph:
            graph.update({node: {first + 1: end}})
        else:
            graph[node].update({first + 1: end})
    return graph


def graph_skip_function(data, skip_n=1):
    '''

    Args: 后向切片
        data: 序列数据
        skip_n: 跳的步数

    Returns: 后向切片

    '''
    DPG = {}
    for i in range(len(data) - skip_n):
        node = data[i]
        first, end = i + 1, i + skip_n + 1
        if node not in DPG:
            DPG.update({node: [data[first:end]]})
        else:
            DPG[node].append(data[first:end])
    return DPG


def graph_DPG_function(data, skip_n=1):
    '''

    Args: 有向概率图
        data: 序列数据
        skip_n: 跳的步数

    Returns: 有向概率图

    '''
    graph = {}
    for i in range(len(data) - skip_n):
        node = data[i]
        first, end = i + 1, i + skip_n + 1
        if node not in graph:
            graph.update({node: [data[first:end]]})
        else:
            graph[node].append(data[first:end])
    DPG = {}
    for node, list_ in graph.items():
        temp_lenghts = {}
        temp_n = len(list_)
        for layer in list_:
            layer_to_tuple = ''.join(layer)
            if layer_to_tuple in temp_lenghts:
                temp_lenghts[layer_to_tuple] += 1
                temp_lenghts[layer_to_tuple]
            else:
                temp_lenghts.update({layer_to_tuple: 1})
        DPG.update({node: {key: value / temp_n for key, value in temp_lenghts.items()}})
    return DPG


def probability_transition_matrix_function(data, skip_n=1):
    '''
    Args:
        data:
        skip_n:

    Returns:
    '''
    graph = {}
    ylab = []
    xlab = []
    for i in range(len(data) - skip_n):
        node = data[i]
        first, end = i + 1, i + skip_n + 1
        layer = data[first:end]
        if node not in graph:
            graph.update({node: [layer]})
        else:
            graph[node].append(layer)
    planning = [len(graph[node]) for node in graph]
    m,n = len(planning),max(planning)
    init_zero = np.zeros((m,n))


    return m,n,init_zero,graph



if __name__ == '__main__':
    # print(graph_DAG_function(word_list, 2))
    # print(graph_skip_function(word_list, 2))
    # print(graph_DPG_function(word_list, skip_n=1))
    print(probability_transition_matrix_function(word_list, 2))
