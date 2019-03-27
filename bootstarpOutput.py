'''
改进后fasttext批量导出数据
注：时间较长，远长于原始fasttext
'''
import os
import fastText.FastText as ff
import time
import linecache

filename="data/"

str_data_path=os.path.join(filename,"train_predict.tsv")#在这里放入商品路径,注意修改文件名
outstr=''

#分词
import jieba
import re

jieba.load_userdict(os.path.join(filename,"dictionary.txt"))#载入词典
stopwords_file = os.path.join(filename,"stop_words.txt")

stop_f = open(stopwords_file,"r",encoding='utf-8')
stop_words = list()
for line in stop_f.readlines():
    line = line.strip()
    if not len(line):
        continue
    stop_words.append(line)
stop_f.close ()

f = open(str_data_path,"r",encoding='utf-8')
result = list()
line_count=0
line_index=[]
for line in f.readlines():
    line_count+=1
    line = line.strip()
    if not len(line):
        line_index.append(line_count)
        continue
    outstr = ''
    line=''.join(line.split())
    line = re.sub(r'[A-Za-z0-9]|/d+', '', line)#去除英文与数字
    if not len(line):
        line_index.append(line_count)
        continue
    seg_list = jieba.cut(line,cut_all=False,HMM=True)#使用jieba分词，精确模式，使用HMM模型识别新词
    for word in seg_list:#去除停用词
        if word not in stop_words:
            if word != '\t':
                outstr += word
                outstr += " "
    if not len(outstr.strip()):
        line_index.append(line_count)
        continue
    result.append(outstr.strip())
f.close()

with open(os.path.join(filename,"trainF.tsv"),"w",encoding='utf-8') as fw:#分好词的处理好的商品描述
    for sentence in result:
        sentence.encode('utf-8')
        data=sentence.strip()
        if len(data)!=0:
            fw.write(data)
            fw.write("\n")

str_data_path_count=0#计算预测文件行数
with open(str_data_path,'r',encoding="utf-8") as f_sum:
    for l in f_sum:
        str_data_path_count+=1

sum_label_finish=list(set(list(range(1, str_data_path_count+1)))-set(line_index))

fo_label=open(os.path.join(filename,"train2.tsv"),"w",encoding="utf-8")#经过处理后的商品
for i in sum_label_finish:
    theline = linecache.getline(str_data_path, i)
    fo_label.write(theline)
fo_label.close()


f_predict_label=open(os.path.join(filename,"bootstarp/test/test_label_predict.tsv"),"w",encoding="utf-8")

import inspect#生成动态变量名
def get_variable_name(variable):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is variable]

d=[]
sum_test=0#计算分词后预测数据集的行数
with open(os.path.join(filename,"trainF.tsv"),'r',encoding="utf-8") as f_test:
    for line_test in f_test:
        sum_test+=1

for d_count in range(0,sum_test):
    d.append({})

#导入模型来预测数据集文件标签
for count in range(0,80):#此处根据上一步训练所用的分类器更改
    prepare_list = globals()
    if __name__ == '__main__':
        for i in range(1, 6):#一次导入五个模型
            prepare_list['classisier' + str(i)] = ff.load_model("data/bootstarp/model/model" + str(count*5+i) + ".model")
    line_nu=0
    with open(os.path.join(filename,"trainF.tsv"), "r", encoding="utf-8") as f_test:
        for line in f_test:
            line = line.strip()
            dict = d[line_nu]
            for i in range(1, 5):
                str_test = prepare_list['classisier' + str(i)].predict(line, k=5, threshold=0.0)
                for j in range(0, 5):
                    # print(str_test[0][j])
                    dict.setdefault(str_test[0][j], 0)
                    dict[str_test[0][j]] += str_test[1][j]
            d[line_nu]=dict
            line_nu+=1

predict=[]
for count_update in d:#综合每个模型选出最终的标签
    dict_update = {}
    for k in sorted(count_update, key=count_update.__getitem__, reverse=True):
                # print(k, dict[k])
        dict_update.setdefault(k, count_update[k])
    Predict_label = list(dict_update.keys())[0].replace("__label__",'')
    #print(Predict_label)
    f_predict_label.write(Predict_label+"\n")
    predict.append(Predict_label)
f_predict_label.close()

p=0
f_train2=open(os.path.join(filename,"train2.tsv"),"w",encoding="utf-8")
with open(str_data_path,"w",encoding="utf-8") as trainf:
    for line in f_train2:
        line=line.strip()
        trainf.write(line+"\t"+predict[p])
        p+=1

os.remove(os.path.join(filename,"train2.tsv"))
os.remove(os.path.join(filename,"trainF.tsv"))

print("end")
