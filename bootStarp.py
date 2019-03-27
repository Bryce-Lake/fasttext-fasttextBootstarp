import os
import re
import random
import linecache
import fastText.FastText as ff
import time
filename = "data/bootstarp/train"

train_max = os.path.join(filename,"train_max.tsv")#多数类文件
train_min = os.path.join(filename,"train_min.tsv")#少数类文件
train_bootstarp = os.path.join(filename,"train_bootstarp.tsv")#用于训练bootstarp不等比例欠采样+fasttext文件,在本程序中生成

c=0.8#欠采样中，多数类与少数类的比例
t=[]
sum_max=0
sum_min=0
with open(train_max,'r',encoding="utf-8") as f_max:
    for line_max in f_max:
        sum_max+=1

with open(train_min,'r',encoding="utf-8")as f_min:
    for line_min in f_min:
        sum_min+=1

for p in range(0,80):#为分类器的个数，注：此处分类器的个数为80个
   t.append(c)
   c+=0.005#比例按照0.005等比增加
   c=round(c,3)

sum_time=0

for i in range(0,80):#80次训练
    f_bootstarp2=open(os.path.join(filename,"train_bootstarp2.tsv"),"w",encoding="utf-8")
    with open(train_bootstarp, "w", encoding="utf-8") as f_bootstarp:
        n = random.randint(int((sum_max+sum_min)/2), int((sum_max+sum_min)*3/5))#此处为每次重采样所用的数据集的容量
        max_count=int(n/2*t[i])
        min_count=n-max_count
        for mx in range(0,max_count):
            theline_max = linecache.getline(train_max, random.randint(1, sum_max))#随机欠采样
            f_bootstarp.write(theline_max)
        for mi in range(0,min_count):
            theline_min = linecache.getline(train_min, random.randint(1, sum_min))#随机采样
            f_bootstarp.write(theline_min)

        result=random.sample(range(1,n+1),n)

        for a in result:
            theline=linecache.getline(train_bootstarp,a)
            f_bootstarp2.write(theline)
    f_bootstarp2.close()

    start=time.time()
    classifier = ff.train_supervised(os.path.join(filename, 'train_bootstarp2.tsv'), dim=64, lr=0.7, wordNgrams=2,
                                     minCount=2, bucket=10000000, label='__label__', thread=20, epoch=7)
    #此处为训练代码
    model = classifier.save_model('data/bootstarp/model/model' + str(i+1) + '.model')  # 保存模型

    end=time.time()

    sum_time+=(end-start)

#此处删除本程序中生成的两个文件，这两个文件仅用来临时作为训练文件，每次运行均生成
os.remove(train_bootstarp)
os.remove(os.path.join(filename,"train_bootstarp2.tsv"))
print("训练所用的时间总和：",sum_time)
print("end")