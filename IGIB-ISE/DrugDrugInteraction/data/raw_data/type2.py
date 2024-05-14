#drugbank_id_1,drugbank_id_2,smiles_2,smiles_1,label
#drugbank_id_1,drugbank_id_2,smiles_2,smiles_1,cid_1,cid_2,label
import pandas as pd 
import numpy as np 
train_data = pd.read_csv("ZhangDDI_train.csv")
valid_data = pd.read_csv("ZhangDDI_valid.csv")
test_data = pd.read_csv("ZhangDDI_test.csv")
train_data = np.array(train_data)
valid_data = np.array(valid_data)
test_data = np.array(test_data)
data = np.concatenate((train_data,valid_data,test_data))
print(data)


import numpy as np

# 假设你的数据存储在变量 data 中，其中每行的前两列是药物1和药物2，第三列是标签
# 请替换下面的示例数据为你的实际数据

# 提取药物1和药物2
drugs1 = data[:, 0]
drugs2 = data[:, 1]

# 获取所有药物的唯一值
unique_drugs = np.unique(np.concatenate((drugs1, drugs2)))

# 随机打乱药物顺序
np.random.shuffle(unique_drugs)

# 定义划分比例（可以根据需求调整）
train_ratio = 0.7
val_ratio = 0.3

# 计算划分点
train_split = int(len(unique_drugs) * train_ratio)
val_split = int(len(unique_drugs) * (train_ratio + val_ratio))

# 划分训练集、验证集和测试集的药物
train_drugs = unique_drugs[:train_split]
val_test_drugs = unique_drugs[train_split:]

# 随机挑选一半的药物放入验证集，一半放入测试集
val_drugs = val_test_drugs[:len(val_test_drugs)]
# 根据药物划分数据
train_set = data[(np.isin(drugs1, train_drugs) & np.isin(drugs2, train_drugs))]
val_set1 = data[(np.isin(drugs1, val_drugs) & np.isin(drugs2, val_drugs))]

val_set = val_set1[:len(val_set1)//2]
test_set = val_set1[len(val_set1)//2:]
# 输出各个集合的大小
print("Train set size:", len(train_set))
print("Validation set size:", len(val_set))
print("Test set size:", len(test_set))
import csv
csv_file_path = 'ZhangDDI_train.csv'

# 使用csv模块写入CSV文件
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # 写入数据
    writer.writerows(train_set)

print(f'CSV文件已保存到: {csv_file_path}')
import csv
csv_file_path = 'ZhangDDI_valid.csv'

# 使用csv模块写入CSV文件
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # 写入数据
    writer.writerows(val_set)

print(f'CSV文件已保存到: {csv_file_path}')

import csv
csv_file_path = 'ZhangDDI_test.csv'

# 使用csv模块写入CSV文件
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # 写入数据
    writer.writerows(test_set)

print(f'CSV文件已保存到: {csv_file_path}')