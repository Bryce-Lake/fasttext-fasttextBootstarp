import os
import random
import re
import linecache
import fastText.FastText as ff


filename="data/"
#filename = "data/bootstarp/train"
line_nu=0
with open(os.path.join(filename,"train.tsv"),"r",encoding="utf-8") as f:
    for line in f:
        line_nu+=1
        if line_nu>=0 and line_nu<5000:
            print(line)
        #if line=="服饰鞋帽--男鞋--休闲鞋":
    print(line_nu)

