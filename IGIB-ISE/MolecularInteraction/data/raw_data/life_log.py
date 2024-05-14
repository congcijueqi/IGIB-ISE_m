import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('data_faguangtuan_data_Lifetime (ns).csv')

if 'Absorption max (nm)' in df.columns:
    # 将值为0的行删除
    df = df[df['Absorption max (nm)'] != 0]
    # 将非0的值取对数
    df['Absorption max (nm)'] = np.log(df['Absorption max (nm)'])

# 保存修改后的数据回CSV文件
df.to_csv('Lifetime.csv', index=False)
