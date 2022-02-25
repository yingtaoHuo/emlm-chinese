import csv

f_train = open('pretrain_data.tsv','r',encoding='utf-8',newline='')
r_train = open('pretrain_data_tsv_split','w+', encoding='utf-8',newline='')
csv_train = csv.reader(f_train)
csv_pretrain = csv.writer(r_train)
num = 0
for item in csv_train:
    csv_pretrain.writerow(item)
    # print(len(item))
    num += 1
    if num > 1000:
        break
# print(num)
# print(len(csv_train))