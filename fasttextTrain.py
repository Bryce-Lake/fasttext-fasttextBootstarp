# -*- coding:utf-8 -*-
'''
原始fasttext训练,且采用十折交叉验证法检验最终准确率
'''
import os
import re
import random
import linecache

filename="data"

train_new=os.path.join(filename,"train_new.tsv")

fileTrain=open(train_new,'r',encoding="utf-8")
#统计文件行数
count=0
for line in fileTrain:
    count+=1

resultList=random.sample(range(1,count+1),count)

#用于平均切分列表，ls表示列表，each表示切分后列表的大小
def divide(ls, each):
    dividedLs = []
    eachExact = float(each)
    groupCount = len(ls) / each
    groupCountExact = len(ls) / eachExact
    start = 0
    for i in range(int(groupCount)):
        dividedLs.append(ls[start:start + each])
        start = start + each
    if (groupCountExact-int(groupCount))>0:  # 假如有余数，将剩余的所有元素加入到最后一个分组
        dividedLs.append(ls[int(groupCount) * each:])
    return dividedLs

result=divide(resultList,50000)

fileTrain.close()

import fastText.FastText as ff
import time
import os
#十折交叉验证法检验最终模型的准确率
precision=[]

for i in range(0,len(result)):
    f_train = open(os.path.join(filename, "original/train.tsv"), 'w', encoding="utf-8")#将训练集分成训练文件与测试文件
    f_test = open(os.path.join(filename, "original/test.tsv"), 'w', encoding="utf-8")#此处自动生成
    a=result[i]
    list_train=random.sample(list(set(resultList)-set(a)),(len(resultList)-len(a)))
    for x in list_train:
        f_train.write(linecache.getline(train_new,x))
    for y in a:
        f_test.write(linecache.getline(train_new,y))
    f_train.close()
    f_test.close()
    start=time.time()
    classifier = ff.train_supervised(os.path.join(filename, 'original/train.tsv'), dim=64, lr=0.7, wordNgrams=2,
                                    minCount=2,bucket=10000000,label = '__label__',thread = 20,epoch=7)#训练代码
    model=classifier.save_model(os.path.join(filename,'original/model/model'+str(i+1)+'.model')) # 保存模型
    test = classifier.test(os.path.join(filename, 'original/test.tsv'), k=1)#测试
    end=time.time()
    precision.append((test[1],end-start))
    print('模型预测准确率：', test[1])
    print("训练时间为：",end-start)

sum_precision=0
sum_time=0
for i,t in precision:
    sum_precision+=i
    sum_time+=t

#用于删除本程序中生成的训练文件，测试文件，这两文件每次运行均会生成，最终删除
os.remove(os.path.join(filename,"original/train.tsv"))
os.remove(os.path.join(filename,"original/test.tsv"))

print("模型平均准确度为:",round(sum_precision/len(result),5))
print("模型平均训练时间：",sum_time/len(result))