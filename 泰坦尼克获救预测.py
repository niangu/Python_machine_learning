#线性回归
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression #线性回归
from sklearn.model_selection import KFold #K折交叉验证
from sklearn.model_selection import train_test_split #划分数据集为测试集和训练集

titanic = pd.read_csv('titanic_train.csv')
#print(titanic.describe())

titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median()) #用中位数填充Age的缺失值
#print(titanic)

#将Sex列的属性数值化
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
#print(titanic["Sex"][:5]) #0 1 1 1 0
#print("------------------------")

#将Enbarkes列的属性数值化
titanic["Embarked"] = titanic["Embarked"].fillna("S") #缺失值填充为最多的值
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
#print(titanic["Embarked"][:10])

#取出数据和标签
columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = titanic[columns]
#print(X[:5])
y = titanic["Survived"]
#print(y[:5])


#用线性回归预测
alg = LinearRegression()

kf = KFold(n_splits=3) #用训练集数据进行交叉验证
kf.get_n_splits(X)

predictions = []
for train, test in kf.split(X):
    train_data = titanic[columns].iloc[train, :]
    train_target = titanic["Survived"].iloc[train]

    alg.fit(train_data, train_target)
    pred_target = alg.predict(titanic[columns].iloc[test, :])
    predictions.append(pred_target)

#print(predictions) #结果是预测的值并不是分类值,一个列表中包括了三个数组（上面操作我们分成三个不同的测试集得出预测结果）

#线性回归的预测值来分类
predictions = np.concatenate(predictions, axis=0) #我们将三个测试集得出的结果进行连接

predictions[predictions > 0.5] = 1 #阈值0.5
predictions[predictions <= 0.5] = 0

print(predictions)
print(len(predictions)) #891

accurary = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
print(accurary) #0.2615039281705948 精确度很低，效果很差


#逻辑回归
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression #线性回归
from sklearn.linear_model import LogisticRegression #逻辑回归
from sklearn.model_selection import KFold #K折交叉验证
from sklearn.model_selection import train_test_split #划分数据集为测试集和训练集

titanic = pd.read_csv('titanic_train.csv')
#print(titanic.describe())

titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median()) #用中位数填充Age的缺失值
#print(titanic)

#将Sex列的属性数值化
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
#print(titanic["Sex"][:5]) #0 1 1 1 0
#print("------------------------")



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression #线性回归
from sklearn.linear_model import LogisticRegression #逻辑回归
from sklearn.model_selection import KFold #K折交叉验证
from sklearn.model_selection import train_test_split #划分数据集为测试集和训练集

titanic = pd.read_csv('titanic_train.csv')
#print(titanic.describe())

titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median()) #用中位数填充Age的缺失值
#print(titanic)

#将Sex列的属性数值化
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
#print(titanic["Sex"][:5]) #0 1 1 1 0
#print("------------------------")

#将Enbarkes列的属性数值化
titanic["Embarked"] = titanic["Embarked"].fillna("S") #缺失值填充为最多的值
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
#print(titanic["Embarked"][:10])

#取出数据和标签
columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = titanic[columns]
#print(X[:5])
y = titanic["Survived"]
#print(y[:5])

#用逻辑回归预测

lr = LogisticRegression()
lr.fit(X, y)
print(lr.score(X, y))  #0.7991021324354658，准确率还可以



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression #线性回归
from sklearn.linear_model import LogisticRegression #逻辑回归
from sklearn.ensemble import RandomForestClassifier #随机森林
from sklearn.model_selection import KFold #K折交叉验证
from sklearn.model_selection import train_test_split #划分数据集为测试集和训练集
from sklearn.model_selection import cross_val_score

titanic = pd.read_csv('titanic_train.csv')
#print(titanic.describe())

titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median()) #用中位数填充Age的缺失值
#print(titanic)

#将Sex列的属性数值化
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
#print(titanic["Sex"][:5]) #0 1 1 1 0
#print("------------------------")

#将Enbarkes列的属性数值化
titanic["Embarked"] = titanic["Embarked"].fillna("S") #缺失值填充为最多的值
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
#print(titanic["Embarked"][:10])

#取出数据和标签
columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = titanic[columns]
#print(X[:5])
y = titanic["Survived"]
#print(y[:5])


#用随机森林回归预测
rfc = RandomForestClassifier(n_estimators=10, random_state=1)

kf = KFold()
kf.get_n_splits(X)

scores = cross_val_score(rfc, X, y, cv=kf)

print(scores.mean()) #0.7856341189674523


#用随机森林回归预测
rfc = RandomForestClassifier(n_estimators=50, min_samples_split=4, min_samples_leaf=2, random_state=1)#n_estimators=50几颗树min_samples_split=4最小切分点min_samples_leaf=2最小叶子数量
#随机森林调数量
kf = KFold()
kf.get_n_splits(X)

scores = cross_val_score(rfc, X, y, cv=kf)

print(scores.mean())  # 0.8159371492704826



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression #线性回归
from sklearn.linear_model import LogisticRegression #逻辑回归
from sklearn.ensemble import RandomForestClassifier #随机森林
from sklearn.model_selection import KFold #K折交叉验证
from sklearn.model_selection import train_test_split #划分数据集为测试集和训练集
from sklearn.model_selection import cross_val_score

titanic = pd.read_csv('titanic_train.csv')
#print(titanic.describe())

titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median()) #用中位数填充Age的缺失值
#print(titanic)

#将Sex列的属性数值化
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
#print(titanic["Sex"][:5]) #0 1 1 1 0
#print("------------------------")

#将Enbarkes列的属性数值化
titanic["Embarked"] = titanic["Embarked"].fillna("S") #缺失值填充为最多的值
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
#print(titanic["Embarked"][:10])

#取出数据和标签
columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = titanic[columns]
#print(X[:5])
y = titanic["Survived"]
#print(y[:5])

#添加一些潜在特征
X["FamilySize"] = titanic["SibSp"] + titanic["Parch"] #添加家庭成员数量信息
X["Namelength"] = titanic["Name"].apply(lambda x : len(x)) #名字的长度

print(X[:5])


#用随机森林回归预测
rfc = RandomForestClassifier(n_estimators=50, min_samples_split=4, min_samples_leaf=2, random_state=1)

kf = KFold()
kf.get_n_splits(X)

scores = cross_val_score(rfc, X, y, cv=kf)

print(scores.mean()) #0.8237934904601572



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression #线性回归
from sklearn.linear_model import LogisticRegression #逻辑回归
from sklearn.ensemble import RandomForestClassifier #随机森林
from sklearn.ensemble import GradientBoostingClassifier #Boosting集成算法
from sklearn.model_selection import KFold #K折交叉验证
from sklearn.model_selection import train_test_split #划分数据集为测试集和训练集
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest,f_classif #最好的特征选择

titanic = pd.read_csv('titanic_train.csv')
#print(titanic.describe())

titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median()) #用中位数填充Age的缺失值
#print(titanic)

#将Sex列的属性数值化
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
#print(titanic["Sex"][:5]) #0 1 1 1 0
#print("------------------------")

#将Enbarkes列的属性数值化
titanic["Embarked"] = titanic["Embarked"].fillna("S") #缺失值填充为最多的值
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
#print(titanic["Embarked"][:10])

#取出数据和标签
columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = titanic[columns]
#print(X[:5])
y = titanic["Survived"]
#print(y[:5])

#添加一些潜在特征
X["FamilySize"] = titanic["SibSp"] + titanic["Parch"] #添加家庭成员数量信息
X["Namelength"] = titanic["Name"].apply(lambda x : len(x)) #名字的长度

#print(X[:5])

kf = KFold(n_splits=3) #交叉验证
kf.get_n_splits(X)

gbc = GradientBoostingClassifier(n_estimators=50,random_state=1)

scores = cross_val_score(gbc ,X , y, cv=kf)

print(scores.mean()) #0.8159371492704827


