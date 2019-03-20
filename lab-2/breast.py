import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('Breast Cancer Coimbra.csv',header=0)


## 1.探索数据的结构
# 查看数据集前5行数据
print(df.head())
# 查看数据集的规模
print(df.shape)				# 查看结果：(116, 10)
# 检查是否存在缺失数据
print(df.isnull().any())	# 检验结果：不存在缺失数据
# 检查是否存在重复数据
print(df.duplicated())		# 检验结果：不存在重复数据
# 查看每个类的样本数
print(df['Classification'].value_counts())


## 2.查看数据的统计特征
print(df.describe())
df.boxplot()	# 画出箱形图
plt.show()


## 3.数据标准化
'''
import numpy as np
df=df.apply(lambda x:(x-np.min(x))/(np.max(x)-np.min(x)))
print(df)
'''
# 正确做法
from sklearn.preprocessing  import StandardScaler 
ss = StandardScaler()
temp = df.iloc[:,:-1].values
df.iloc[:,:-1] = ss.fit_transform(temp)
print(df)


## 4.查看属性的相关性
cov_data=df.corr()
print(cov_data)
image=plt.matshow(cov_data,cmap=plt.cm.Spectral)	# 画出相关性矩阵
plt.colorbar(image,ticks=[-1,0,1])
plt.show()


## 5.用PCA方法降维
from sklearn.decomposition import PCA
# 转换为numpy数组
df_data=df.values
# PCA降维
# 1）设置n_components=2时：
pca=PCA(n_components=2)
X_pca=pca.fit_transform(df_data[:,:9])
print('n_components=2时:',X_pca.shape)
# 查看特征子集的散点图
plt.scatter(X_pca[:,0],X_pca[:,1],c=df_data[:,9],alpha=0.8,edgecolors='none')
plt.show()
# 2）设置n_components=0.8时：
pca=PCA(n_components=0.8)
X_pca_2=pca.fit_transform(df_data)
print('n_components=0.8时:',X_pca_2.shape) 	 
# 3）设置n_components='mle'时：
pca=PCA(n_components='mle')
X_pca_3=pca.fit_transform(df_data)
print("n_components='mle'时:",X_pca_3.shape) 	


## 6.RFECV方法选择合适的特征子集，查看选择的特征数
# 预测变量
X = df.iloc[:,:9]
# 目标变量
Y = df.iloc[:,9]
# 划分训练集与测试集（取0.7，0.3）
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=101)
#使用RFECV选择最优特征子集
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
from sklearn.feature_selection import RFECV
rfecv=RFECV(estimator=lr,step=1,cv=10,scoring='accuracy')
rfecv.fit(X_train,Y_train)
#输出特征数
print("选择的特征数(RFECV）： %d" %rfecv.n_features_)

