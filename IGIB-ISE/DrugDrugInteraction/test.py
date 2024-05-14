#在这里我们将对同一种溶质溶剂进行划分
import pandas as pd
import numpy as np
import csv
data = pd.read_csv("data/raw_data/ZhangDDI_train.csv")
data =np.array(data)
drug_ids_2d1 = data[:, 0]
drug_ids_2d2 = data[:, 1]
result = np.concatenate((drug_ids_2d1, drug_ids_2d2), axis=0)
# 将数组的形状重新整理为 (10000, 2)
#reshaped_drug_data = drug_ids_2d.reshape((len(data), -1))

# 使用 unique 函数找出所有不重复的药物编号
print(result)
unique_drug_ids = np.unique(result)

print("不重复的药物编号：", unique_drug_ids)
ls =[]
for j in range(len(unique_drug_ids)):
    n=[]
    for i in range(len(data)):
        if data[i][0]==unique_drug_ids[j]:
            n.append(i)
    ls.append(n)
    print(j)
print(ls)
file_path = 'output.txt'

with open(file_path, 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerows(ls)