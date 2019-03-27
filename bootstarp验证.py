import logging
import fastText.FastText as ff
import time
import os
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

filename="data"
test_path=os.path.join(filename,"bootstarp/testBootstarp.tsv")#测试集
f_test_product=open(os.path.join(filename,"bootstarp/test/test_product.tsv"),'w',encoding="utf-8")
f_test_label=open(os.path.join(filename,"bootstarp/test/test_label.tsv"),'w',encoding="utf-8")
with open(test_path,'r',encoding="utf-8") as f:
    for line in f:
        the_product = line.split("\t")[0]
        the_label = line.split("\t")[1].replace("__label__", '')
        f_test_product.write(the_product.strip() + "\n")
        f_test_label.write(the_label.strip() + "\n")
f_test_product.close()
f_test_label.close()

test_product=os.path.join(filename,"bootstarp/test/test_product.tsv")#测试集所分离的商品描述
test_label=os.path.join(filename,"bootstarp/test/test_label.tsv")#测试集所用的商品标签


f_predict_label=open(os.path.join(filename,"bootstarp/test/test_label_predict.tsv"),"w",encoding="utf-8")

import inspect#生成动态变量名
def get_variable_name(variable):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is variable]

d=[]
sum_test=0#计算测试集的行数
with open(test_path,'r',encoding="utf-8") as f_test:
    for line_test in f_test:
        sum_test+=1

for d_count in range(0,sum_test):
    d.append({})

#导入模型来预测测试集文件标签
for count in range(0,1):#此处根据上一步训练所用的分类器更改
    prepare_list = globals()
    if __name__ == '__main__':
        for i in range(1, 6):#一次导入五个模型
            prepare_list['classisier' + str(i)] = ff.load_model("data/bootstarp/model/model" + str(count*5+i) + ".model")
    line_nu=0
    with open(test_product, "r", encoding="utf-8") as f_test:
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

for count_update in d:#综合每个模型选出最终的标签
    dict_update = {}
    for k in sorted(count_update, key=count_update.__getitem__, reverse=True):
                # print(k, dict[k])
        dict_update.setdefault(k, count_update[k])
    Predict_label = list(dict_update.keys())[0].replace("__label__",'')
    #print(Predict_label)
    f_predict_label.write(Predict_label+"\n")
f_predict_label.close()

#以下为计算最终的准确率
f1=open(os.path.join(filename,"bootstarp/test/test_label.tsv"),"r",encoding="utf-8")
f2=open(os.path.join(filename,"bootstarp/test/test_label_predict.tsv"),"r",encoding="utf-8")

l=[]
test_label_count=0
sum=0
for line_test_label in f1:
    l.append(line_test_label.strip())

for j in f2:
    j=j.strip()
    if j== l[test_label_count]:
        sum+=1
    test_label_count+=1

f1.close()
f2.close()
print("fasttext结合bootstarp不等比例欠采样的准确率为: " , round(sum/150000,5))