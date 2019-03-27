'''
数据预处理
分词部份
'''
#整理停用词文本文件
#encoding=utf-8
import os

filename = "data"
stopWordsPath=os.path.join(filename,"stop.txt")#原始停用词字典

trainProductPath = os.path.join(filename,"original/train_product.tsv")#训练集，商品描述地址
# 若要修改分词文件在此修改

#去除停用词空行
with open(os.path.join(filename,"stop.txt"),"r",encoding='utf-8') as f:
    result=list()
    for line in f:
        line=line.strip()
        if not len(line):
            continue
        result.append(line)

with open(os.path.join(filename,"stop_words.txt"),"w",encoding='utf-8') as fw:
    for sentence in result:
        sentence.encode('utf-8')
        data=sentence.strip()
        if len(data)!=0:
           fw.write(data)
           fw.write("\n")


#最终分词过程，包括去掉英文，数字，停用词，及其他特殊符号
#仅仅保留中文
import jieba
import re
import linecache
import  jieba.posseg as pseg

jieba.load_userdict(os.path.join(filename,"dictionary.txt"))#载入词典

stopwords_file = os.path.join(filename,"stop_words.txt")#经处理后的停用词字典

stop_f = open(stopwords_file,"r",encoding='utf-8')
stop_words = list()
for line in stop_f.readlines():
    line = line.strip()
    if not len(line):
        continue
    stop_words.append(line)
stop_f.close ()

print("停用词字典中停用词个数:",len(stop_words))

f = open(trainProductPath,"r",encoding='utf-8')
result = list()
line_count=0
line_index=[]
for line in f.readlines():
    line_count+=1
    line = line.strip()
    if not len(line):
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
#print(line_count)
#print(line_index)

with open(os.path.join(filename,"original/train_product2.tsv"),"w",encoding='utf-8') as fw:#处理好的商品描述
    for sentence in result:
        sentence.encode('utf-8')
        data=sentence.strip()
        if len(data)!=0:
            fw.write(data)
            fw.write("\n")

sum_train_product=0
with open(trainProductPath,'r',encoding="utf-8") as f_sum_train_product:
    for line_trainProduct in f_sum_train_product:
        sum_train_product+=1

sum_label_finish=list(set(list(range(1, sum_train_product+1)))-set(line_index))

fo_label=open(os.path.join(filename,"original/train_label2.tsv"),"w",encoding="utf-8")#经过处理后的标签
for i in sum_label_finish:
    theline = linecache.getline(os.path.join(filename,"original/train_label.tsv"), i)
    fo_label.write(theline)
fo_label.close()


train_product=open(os.path.join(filename,"original/train_product2.tsv"),"r",encoding="utf-8")#分词后的训练集商品描述
train_label=open(os.path.join(filename,"original/train_label2.tsv"),"r",encoding="utf-8")#经过处理的训练集标签

#将处理后的训练集商品描述与商品标签组合成最终文件
with open(os.path.join(filename,"train_new.tsv"),"w",encoding="utf-8")as train_new:
    line = ""

    for line_product in train_product:
        for line_label in train_label:
            line_product = line_product.replace("\n", '')
            line = line_product + "\t"+"__label__" + line_label#fasttext要求的数据形式为：商品描述+__label__+商品标签
            train_new.write(line)
            break

train_product.close()
train_label.close()

#此处删除本程序中生成的两个文件，后续无用
os.remove(os.path.join(filename,"original/train_product2.tsv"))
os.remove(os.path.join(filename,"original/train_label2.tsv"))

print ("end")