## 由于编码问题，将csv文件改为UTF-8编码


## 一、film-1.csv
import pandas as pd
df1=pd.read_csv('film-1.csv',header=0)
# 删除空行
df1.dropna(how='all',inplace=True)
# 1)统一影片的类型
df1.replace('美国 科幻/动作','美国 科幻',inplace=True)
# 2)删除重复数据行
df1.drop_duplicates(inplace=True)
# 3)影片类型列分割为国家和类型两列
df1['国家']=df1['类型'].str.split(expand=True)[0]
df1['类型']=df1['类型'].str.split(expand=True)[1]


## 二、film-2.csv
from numpy import nan as NA
import numpy as np
df2=pd.read_csv('film-2.csv',header=1)
# 原文件中表示空值的'--'转换为NA
df2.replace('--',NA,inplace=True)
# 1)删除全为空的列
df2.dropna(how='all',axis=1,inplace=True)
# 2)票房缺失值用0填充
df2.fillna(0,inplace=True)
# 3)统一'哈票票房'中的单位到万元
rows_with_yuan = df2['哈票票房（万元）'].str.contains('元').fillna(False)
for i,yuan_row in df2[rows_with_yuan].iterrows():  #使用迭代器
    piaofang=float(yuan_row['哈票票房（万元）'][:-1])/10000 
    df2.at[i,'哈票票房（万元）'] = '{}'.format(round(piaofang,2))
# 4)添加'票房合计'列，计算每部片子的每天票房的合计
df2_temp=df2.iloc[:,2:]		# 取出df2中四列票房数据，存入df2_temp中
df2_temp=pd.DataFrame(df2_temp,dtype=np.float)		# 将df2_temp每列的数据类型设置为float，方便求和计算
df2['票房合计']=df2_temp.apply(lambda x: x.sum(),axis=1)


## 三、filmout.csv
# 1)合并前面两个数据集
df3=pd.merge(df1,df2,on='影片名')
# 2)结果写入filmout.csv，行索引不写入文件
df3.to_csv('filmout.csv',mode='w',index=False)
# 3)统计每部片子的总票房
grouped=df3.groupby(df3['影片名'])
print(grouped)
print(grouped['票房合计'].sum())
