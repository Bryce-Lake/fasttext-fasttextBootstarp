'''
该文件用于bootstarp结合不等比例欠采样分离训练集，测试集
'''

#!/usr/bin/python
import os
import re
import random
import linecache
# -*- coding:utf-8 -*-

filename = "data"

sum_train_new=0

with open(os.path.join(filename,"train_new.tsv"),'r',encoding="utf-8") as f:
    for line in f:
        sum_train_new+=1

sum_test=int(sum_train_new*0.3)
sum_train=sum_train_new - sum_test

fo =open(os.path.join(filename,"train_new.tsv"),"r",encoding="utf-8")
fo_test=open(os.path.join(filename,"bootstarp/testBootstarp.tsv"),"w",encoding="utf-8")
fo_train=open(os.path.join(filename,"bootstarp/trainBootstarp.tsv"),"w",encoding="utf-8")

test_List=random.sample(range(1,sum_train_new+1),sum_test)
train_List=random.sample(range(1,sum_train_new+1),sum_train)

for a in test_List:
    theline = linecache.getline(os.path.join(filename,"train_new.tsv"), a)
    fo_test.write(theline)

for i in train_List:
    theline_train=linecache.getline(os.path.join(filename,"train_new.tsv"), i)
    fo_train.write(theline_train)


fo.close()
fo_test.close()
fo_train.close()