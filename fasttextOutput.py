'''
原始fasttext导入模型，批量预测数据
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

classifier = ff.load_model(os.path.join(filename,'original/model/model1.model'))

label=[]
with open(os.path.join(filename,"trainF.tsv"),'r',encoding="utf-8") as f:
    for lin in f:
        lin=lin.strip()
        test = classifier.predict(lin, k=1)
        #print(test)
        t=str(test[0]).replace("__label__",'')
        t=t.replace(",",'').replace("(",'').replace(")",'').replace("'",'').replace("'",'')
        label.append(t)


i=0
f1=open(os.path.join(filename,"train2.tsv"),'r',encoding="utf-8")

with open(str_data_path,'w',encoding="utf-8") as f_train:
    for line in f1:
        f_train.write(line.strip()+"\t"+label[i]+"\n")
        i+=1

f1.close()

os.remove(os.path.join(filename,"train2.tsv"))
os.remove(os.path.join(filename,"trainF.tsv"))
print("end")


