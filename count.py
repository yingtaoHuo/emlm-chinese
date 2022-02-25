import csv
train_data = []
f_train = open('web_zh_2019','r',encoding='utf-8',newline='')
csv_train = csv.reader(f_train)
for item in csv_train:
    if len(item) == 0:
        continue
    train_data.append(item[0])
print(len(train_data))
