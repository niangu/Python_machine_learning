import pandas as pd
import numpy as np
import scipy.stats as ss
import seaborn as sns
sns.set_context(context="poster",font_scale=1.2)
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA
#向下根据部门钻取
#sns.barplot(x="salary",y="left"hue="department",data=df)
#plt.show()
#连续值
#sl_s = df["satisfaction_level"]
#sns.barplot(list(range(len(sl_s))),sl_s.sort_values())
#plt.show()
#离散值相关性分析
sns.heatmap(df.corr(),vmin=1,vmax=1,cmap=sns.color_palette("RdBu",n_colors=128))
plt.show()


#Gini
def getGini(a1, a2):
    assert (len(a1) == len(a2))
    d = dict()
    for i in list(range(len(a1))):
        d[a1[i]] = d.get(a1[i], []) + [a2[i]]
    return 1 - sum([getProbSS(d[k]) * len(d[k]) / float(len(a1)) for k in d])
#可能性平方和
def getProbSS(s):
    if not isinstance(s,pd.core.series.Series):
        s = pd.Series(s)
    prt_ary = np.array(pd.groupby(s, by=s).count().values / float(len(s)))
    return sum(prt_ary ** 2)
#熵
def getEntropy(s):
    if not isinstance(s, pd.core.series.Series):#判断是不是一个Series
        s = pd.Series(s)#如果不是则转换为Series
    prt_ary = np.array(pd.groupby(s, by=s).count().values / float(len(s)))
    return -(np.log2(prt_ary) * prt_ary).sum()
#条件熵
def getCondEntropy(a1, a2):
    assert (len(a1) == len(a2))
    d = dict()
    for i in list(range(len(a1))):
        d[a1[i]] = d.get(a1[i], []) + [a2[i]]
    return sum([getEntropy(d[k]) * len(d[k]) / float(len(a1)) for k in d])
#熵增益
def getEntropyGain(a1, a2):
    return getEntropy(a2) - getCondEntropy(a1, a2)
#熵增益率
def getEntropyGainRatio(a1, a2):
    return getEntropyGain(a1, a2) / getEntropy(a2)
#相关度
def getDiscreteRelation(a1, a2):
    return getEntropyGain(a1, a2) / math.sqrt(getEntropy(a1) * getEntropy(a2))

def main():
    df=pd.read_csv("./data/HR.csv")
    #相关图
    sns.heatmap(df.corr())
    sns.heatmap(df.corr(), vmax=1, vmin=-1)
    plt.show()
    #PCA降维
    my_pca=PCA(n_components=7)
    lower_mat=my_pca.fit_transform(df.drop(labels=["salary","department","left"],axis=1).values)
    print(my_pca.explained_variance_ratio_)
    sns.heatmap(pd.DataFrame(lower_mat).corr())
    plt.show()
    #离散相关性度量
    s1 = pd.Series(["X1", "X1", "X2", "X2", "X2", "X2"])
    s2 = pd.Series(["Y1", "Y1", "Y1", "Y2", "Y2", "Y2"])
    print(getEntropy(s1))
    print(getEntropy(s2))
    print(getCondEntropy(s1, s2))
    print(getCondEntropy(s2, s1))
    print(getEntropyGain(s1, s2))
    print(getEntropyGain(s2, s1))
    print(getEntropyGainRatio(s1, s2))
    print(getEntropyGainRatio(s2, s1))
    print(getDiscreteRelation(s1, s2))
    print(getDiscreteRelation(s2, s1))

if __name__=="__main__":
    main()
