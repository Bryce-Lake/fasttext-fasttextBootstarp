'''
将原始数据分离得到商品描述，商品标签
'''
import os
import random
import re
import linecache

#放入的原始数据应位于data/中
filename = "data"

f_product=open(os.path.join(filename, "original/train_product.tsv"),"w",encoding="utf-8")
f_label=open(os.path.join(filename, "original/train_label.tsv"),"w",encoding="utf-8")
with open(os.path.join(filename,"train.tsv"),"r",encoding="utf-8") as f:#在此处注意修改原始数据名，此处为
      for line in f:
          the_product=line.split("\t")[0]
          the_label=line.split("\t")[1]
          f_product.write(the_product.strip()+"\n")
          f_label.write(the_label.strip()+"\n")

f_product.close()
f_label.close()