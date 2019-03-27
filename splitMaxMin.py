'''
将训练集分成多数类与少数类，为不等比例欠采样做准备
'''
import os
import re
import random
import linecache
filename = "data/bootstarp/train"

#将处理好的训练数据分割成商品描述与商品标签
train_new="data/bootstarp/trainBootstarp.tsv"
train_product=open(os.path.join(filename,"train_product.tsv"),"w",encoding="utf-8")
train_label=open(os.path.join(filename, "train_label.tsv"),"w",encoding="utf-8")
with open(train_new, "r", encoding="utf-8") as f:
    for line in f:
        the_product = line.split("\t")[0]
        the_label = line.split("\t")[1].replace("__label__", '')
        train_product.write(the_product.strip() + "\n")
        train_label.write(the_label.strip() + "\n")

train_product.close()
train_label.close()

fo =open(os.path.join(filename,"train_label.tsv"),"r",encoding="utf-8")
foo=open(os.path.join(filename,"train_product.tsv"),"r",encoding="utf-8")

list_count=[]
line_nu = 0#计数器
for line in fo:#一行行的把数据从硬盘加载到内 存里读出来,读取一定行数
    line_nu += 1
    line=line.strip()
    if line == "服饰鞋帽--男鞋--休闲鞋" or line == "手机数码--手机配件--手机保护套" \
            or line == "家居家装--家纺--床品套件" or line == "汽配用品--汽车装饰--车身装饰件" \
            or line == "钟表礼品--礼品--工艺礼品" or line == "汽配用品--汽车装饰--座垫" \
            or line == "汽配用品--汽车装饰--功能小件" or line == "汽配用品--汽车装饰--脚垫" \
            or line == "服饰鞋帽--女鞋--凉鞋" or line == "箱包皮具--潮流女包--单肩包" \
            or line == "服饰鞋帽--女鞋--单鞋" or line == "服饰鞋帽--女鞋--休闲鞋" \
            or line == "钟表礼品--礼品--礼品文具" or line == "钟表礼品--礼品--创意礼品" \
            or line == "母婴用品/玩具乐器--毛绒布艺--毛绒/布艺" or line == "珠宝饰品--翡翠玉石--项链/吊坠" \
            or line == "家装建材--灯饰照明--吸顶灯" or line == "服饰鞋帽--女鞋--女靴" \
            or line == "运动户外--户外装备--便携桌椅床" or line == "箱包皮具--潮流女包--双肩包" \
            or line == "钟表礼品--钟表--国表" or line == "家居家装--家装软饰--手工/十字绣" \
            or line == "家居家装--家装软饰--墙贴/装饰贴" or line == "电脑/办公--电脑配件--笔记本配件" \
            or line == "汽配用品--汽车装饰--车衣" or line == "家居家装--家纺--被子" \
            or line == "汽配用品--维修保养--滤清器" or line == "手机数码--手机配件--手机贴膜" \
            or line == "运动户外--户外鞋服--冲锋衣裤" or line == "服饰鞋帽--女鞋--拖鞋/人字拖" \
            or line == "汽配用品--汽车装饰--座套" or line == "母婴用品/玩具乐器--寝居服饰--婴儿外出服" \
            or line == "汽配用品--维修保养--车灯" or line == "钟表礼品--礼品--绿植" \
            or line == "服饰鞋帽--女鞋--高跟鞋" or line == "汽配用品--维修保养--雨刷" \
            or line == "母婴用品/玩具乐器--益智玩具--早教启智" or line == "母婴用品/玩具乐器--妈妈专区--孕妈装" \
            or line == "箱包皮具--精品男包--男士钱包" or line == "厨具锅具--厨房配件--储物/置物架" \
            or line == "服饰鞋帽--男装--T恤" or line == "珠宝饰品--时尚饰品--项链" \
            or line == "珠宝饰品--水晶玛瑙--手镯/手链/脚链" or line == "钟表礼品--礼品--鲜花" \
            or line == "厨具锅具--茶具/咖啡具--整套茶具" or line == "珠宝饰品--时尚饰品--戒指" \
            or line == "钟表礼品--礼品--火机烟具" or line == "母婴用品/玩具乐器--积木拼插--积木" \
            or line == "汽配用品--美容清洗--补漆笔" or line == "箱包皮具--功能箱包--拉杆箱" \
            or line == "厨具锅具--水具酒具--保温杯" or line == "图书杂志--童书--儿童文学" \
            or line == "汽配用品--汽车装饰--后备箱垫" or line == "运动户外--户外装备--背包" \
            or line == "珠宝饰品--时尚饰品--发饰/发卡" or line == "厨具锅具--水具酒具--酒杯/酒具" \
            or line == "母婴用品/玩具乐器--模型玩具--仿真模型" or line == "厨具锅具--厨房配件--厨房DIY/小工具" \
            or line == "运动户外--游泳用品--女士泳衣" or line == "箱包皮具--功能箱包--书包" \
            or line == "厨具锅具--茶具/咖啡具--茶壶" or line == "运动户外--体育用品--羽毛球" \
            or line == "母婴用品/玩具乐器--动漫玩具--卡通周边" or line == "家装建材--灯饰照明--吊灯" \
            or line == "家居家装--家纺--毯子" or line == "珠宝饰品--时尚饰品--耳饰" \
            or line == "家居家装--家装软饰--花瓶花艺" or line == "家居家装--家纺--凉席" \
            or line == "汽配用品--汽车装饰--方向盘套" or line == "服饰鞋帽--内衣--睡衣/家居服" \
            or line == "电脑/办公--外设产品--线缆" or line == "运动户外--体育用品--轮滑滑板" \
            or line == "厨具锅具--茶具/咖啡具--茶杯" or line == "珠宝饰品--水晶玛瑙--项链/吊坠" \
            or line == "钟表礼品--奢侈品--箱包" or line == "家装建材--电工电料--开关插座" \
            or line == "服饰鞋帽--男装--夹克" or line == "电脑/办公--办公设备--安防监控" \
            or line == "汽配用品--维修保养--刹车片/盘" or line == "珠宝饰品--时尚饰品--手链/脚链" \
            or line == "运动户外--户外装备--户外照明" or line == "汽配用品--车载电器--行车记录仪" \
            or line == "运动户外--垂钓用品--钓鱼配件" or line == "箱包皮具--精品男包--单肩/斜挎包" \
            or line == "运动户外--运动鞋包--跑步鞋" or line == "手机数码--影音娱乐--音箱/音响" \
            or line == "食品/饮料/酒水--休闲食品--饼干蛋糕" or line == "医药保健--护理护具--隐形眼镜" \
            or line == "运动户外--户外装备--帐篷/垫子" or line == "母婴用品/玩具乐器--寝居服饰--家居床品" \
            or line == "运动户外--健身训练--运动护具" or line == "电脑/办公--文具/耗材--笔类" \
            or line == "汽配用品--安全自驾--摩托车装备" or line == "运动户外--体育用品--高尔夫" \
            or line == "厨具锅具--茶具/咖啡具--茶具配件" or line == "运动户外--户外装备--望远镜" \
            or line == "珠宝饰品--时尚饰品--饰品配件":
        list_count.append(line_nu)
        print(line)
print(line_nu)


with open(os.path.join(filename,"train_max.tsv"),"w",encoding="utf-8")as f_max:
    for a in list_count:
        theline_product = linecache.getline(os.path.join(filename,"train_product.tsv"), a).strip()
        theline_label = linecache.getline(os.path.join(filename,"train_label.tsv"), a)
        theline = theline_product+"\t"+"__label__"+theline_label
        f_max.write(theline)

sum_train_new=0
with open(train_new,'r',encoding="utf-8") as f_train_new:
    for line_train_new in f_train_new:
        sum_train_new+=1

list_min=list(set(range(1,sum_train_new+1))-set(list_count))

with open(os.path.join(filename,"train_min.tsv"),"w",encoding="utf-8") as f_min:
    for a in list_min:
        theline_product = linecache.getline(os.path.join(filename, "train_product.tsv"), a).strip()
        theline_label = linecache.getline(os.path.join(filename, "train_label.tsv"), a)
        theline = theline_product + "\t" + "__label__" + theline_label
        f_min.write(theline)


fo.close()
foo.close()
