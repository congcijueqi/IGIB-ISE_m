#drugbank_id_1,drugbank_id_2,smiles_2,smiles_1,label
import pandas as pd 
import numpy as np 
train_data = pd.read_csv("ZhangDDI_train.csv")
valid_data = pd.read_csv("ZhangDDI_valid.csv")
test_data = pd.read_csv("ZhangDDI_test.csv")
train_data = np.array(train_data)
valid_data = np.array(valid_data)
test_data = np.array(test_data)
data = np.concatenate((train_data,valid_data,test_data))
idx = np.random.permutation(len(data))
train_idx = idx[0:190000]
valid_idx = idx[190000:260000]
test_idx = idx[260000:310000]
train_data = data[train_idx]
valid_data = data[valid_idx]
test_data = data[test_idx]
import csv
csv_file_path = 'ZhangDDI_train.csv'

# 使用csv模块写入CSV文件
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # 写入数据
    writer.writerows(train_data)

print(f'CSV文件已保存到: {csv_file_path}')
import csv
csv_file_path = 'ZhangDDI_valid.csv'

# 使用csv模块写入CSV文件
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # 写入数据
    writer.writerows(valid_data)

print(f'CSV文件已保存到: {csv_file_path}')
import csv
csv_file_path = 'ZhangDDI_test.csv'

# 使用csv模块写入CSV文件
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # 写入数据
    writer.writerows(test_data)

print(f'CSV文件已保存到: {csv_file_path}')