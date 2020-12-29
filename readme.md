# path_geeneration.py 是路径生成的初始化脚本，请在切换环境时首先运行

## 每次新建模型文件时候运行 project_init.py文件
## 在文件这里设置文件的静态配置 



# 这个是模型保存的文件夹
model_data_path = path.expanduser("~/Documents/算法笔记/000_数据分析demo/拉勾网数据分析/StaticCache/model_data/")

# 这个是所有静态缓存文件的根目录
static_cache = path.expanduser("~/Documents/算法笔记/000_数据分析demo/拉勾网数据分析/StaticCache/")

# 这里是所有的文字 one hot encoding 的模型保存位置
one_hot_encoding_path = "{}{}".format(static_cache,"onehots/")

# 这里是所有的模型路径的保存js日志
model_path_json_file = "{}{}".format(static_cache,"model_path.json")

# 这里是自然语言的问答表所在位置
csvfile_path = "{}{}".format(static_cache,"linux_cmd.csv")

# 这里是自然语言问答表的json文件保存所在的位置
linux_cmd_json = "{}{}".format(static_cache,"linux_cmd.json")

# 这里是需要初始化建模处理的自然语言列名
natrule_language_column_name = "文字说明"

# 这里是返回体的列名
return_body = "命令"

# 这里返回的是系统信息
system_info = "系统"


# 搜索模块 
## 需要运行搜索模块的时候，运行search_model.py
## 搜索函数的入口是 search_engine_function 这个函数


# SendEmail模块是发送邮件提醒的组件
# 抬头的 default_key 参数为默认的邮箱授权码位置




