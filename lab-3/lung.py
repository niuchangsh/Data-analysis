import pandas as pd
from numpy import nan as NA
df=pd.read_csv('lung-cancer.csv',header=None)


## 1.缺失数据后向填充
df.replace('?',NA,inplace=True)			# 原文件中表示空值的'?'转换为NA
df.fillna(method='bfill',inplace=True)


## 2.数据集分割训练集、测试集（75%，25%）
from sklearn.cross_validation import train_test_split 
# 预测变量
X = df.iloc[:,1:]
# 目标变量
Y = df.iloc[:,0]
# 分割训练集、测试集
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=101)


## 3.用Filter方法选择2,4,5,10,20个属性，使用决策树算法，查看不同情况下，测试集的分类精度
# 1）用Filter方法选择2个属性
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
selector_chi2 = SelectKBest(chi2, k=2)   # 用chi2选择2个属性，改变k的值，即可选择4,5,10,20个等不同个数的属性
X_new=selector_chi2.fit_transform(X,Y)
X_train_new,X_test_new,Y_train,Y_test=train_test_split(X_new,Y,test_size=0.25,random_state=101)

# 2）使用决策树算法查看测试集的分类精度
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()
clf=clf.fit(X_train_new,Y_train)
print ("Accuracy of clf Classifier 1:", clf.score(X_test_new, Y_test))


## 4.使用嵌入法（分类算法使用决策树），查看选择的特征数以及测试集的分类精度
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()
clf=clf.fit(X_train,Y_train)
from sklearn.feature_selection import SelectFromModel
model=SelectFromModel(clf,prefit=True)
X_train_1=model.transform(X_train)
X_test_1=model.transform(X_test)
# 查看选择的特征数
print("训练集规模：",X_train_1.shape)		# 从X_train_1.shape的列数中可查看选择的特征数
print("测试集规模：",X_test_1.shape)
# 测试集的分类精度
clf=clf.fit(X_train_1,Y_train)
print ("Accuracy of clf Classifier 2:", clf.score(X_test_1, Y_test))