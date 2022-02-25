import  xml.dom.minidom
import csv
import pandas as pd

# 1. 创建文件对象
f_pretrain = open('pretrain_data.tsv','w+',encoding='utf-8',newline='')
f_train = open('train_data.tsv','w+',encoding='utf-8',newline='')
f_test = open('test_data.tsv','w+',encoding='utf-8',newline='')

# 2. 基于文件对象构建 csv写入对象
csv_pretrain = csv.writer(f_pretrain)
csv_train = csv.writer(f_train)
csv_test = csv.writer(f_test)
# csv_writer.writerow(['文本'])

#打开xml文档
dom_train = xml.dom.minidom.parse('train.xml')
dom_test = xml.dom.minidom.parse('test.xml')

#得到文档元素对象
root = dom_train.documentElement
wb=dom_train.getElementsByTagName('weibo')

for i in range(len(wb)):
    wbi=wb[i]
    sens=wbi.getElementsByTagName('sentence')
    for j in range(len(sens)):
        senj=sens[j]
        if senj.firstChild is None:
            continue
        if senj.hasAttribute('emotion-1-type'):
            em=senj.getAttribute('emotion-1-type')
        else:
            em='none'
        text=senj.firstChild.data
        csv_pretrain.writerow([text,em])
        csv_train.writerow([text,em])

root = dom_test.documentElement
wb=dom_test.getElementsByTagName('weibo')

for i in range(len(wb)):
    wbi=wb[i]
    sens=wbi.getElementsByTagName('sentence')
    for j in range(len(sens)):
        senj=sens[j]
        if senj.firstChild is None:
            continue
        if senj.hasAttribute('emotion-1-type'):
            em=senj.getAttribute('emotion-1-type')
        else:
            em='无'
        text=senj.firstChild.data
        csv_pretrain.writerow([text,em])
        csv_test.writerow([text,em])
        
f_pretrain.close()
f_train.close()
f_test.close()