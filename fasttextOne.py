'''
fasttext识别单个商品
'''

import fastText.FastText as ff
import time
import os

filename="data"
str_data=input("请输入商品名称：")
print("你输入的内容是：",str_data)
outstr=''
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

str_data=re.sub(r'[A-Za-z0-9]|/d+', '', str_data)#去除英文与数字
str_data=jieba.cut(str_data,cut_all=False,HMM=True)

for word in str_data:
    if word not in stop_words:
        if word != '\t':
            outstr += word
            outstr += " "

classifier = ff.load_model(os.path.join(filename,'model/model1.model'))
test = classifier.predict(outstr,k=1,threshold=0.5)
print("输入商品的标签为：",str(test[0]).replace("__label__",''))
print("商品属于该标签的概率为：",test[1])