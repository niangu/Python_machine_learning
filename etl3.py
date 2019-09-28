import pandas as pd
import numpy as np
import scipy.stats as ss
import seaborn as sns
sns.set_context(context="poster",font_scale=1.2)
import matplotlib.pyplot as plt

def main():
    lst=[6,8,10,15,16,24,25,40,67]
    #离散化
    binings,bins=pd.qcut(lst,q=3,retbins=True)#等深分箱
    print(list(bins))
    print(pd.cut(lst,bins=3))#等宽分箱
    print(pd.cut(lst,bins=4,labels=["low","medium","high","very high"]))

    #归一化与标准化
    from sklearn.preprocessing import MinMaxScaler,StandardScaler
    print(MinMaxScaler().fit_transform(np.array([1,4,10,15,21]).reshape(-1,1)))#reshape(-1,1)1列，归一化
    print(StandardScaler().fit_transform(np.array([1,1,1,1,0,0,0,0]).reshape(-1,1)))
    print(StandardScaler().fit_transform(np.array([1, 0, 0, 0, 0, 0, 0, 0]).reshape(-1, 1)))

    #标签化与独热编码
    from sklearn.preprocessing import LabelEncoder,OneHotEncoder
    print(LabelEncoder().fit_transform(np.array(["Down","Down","Up","Down","Up"]).reshape(-1,1)))
    print(LabelEncoder().fit_transform(np.array(["Low","Medium","Low","High","Medium"]).reshape(-1,1)))
    lb_encoder=LabelEncoder()#独热编码，开始
    lb_encoder=lb_encoder.fit(np.array(["Red","Yellow","Blue","Green"]))
    lb_trans_f=lb_encoder.transform(np.array(["Red","Yellow","Blue","Green"]))
    oht_enoder=OneHotEncoder().fit(lb_trans_f.reshape(-1,1))
    print(oht_enoder.transform(lb_encoder.transform(np.array(["Red","Blue"])).reshape(-1,1)).toarray())

    #规范化
    from sklearn.preprocessing import Normalizer
    print(Normalizer(norm="l1").fit_transform([1,1,3,-1,2]))

    #LDA降维
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    y = np.array([0, 0, 0, 1, 1, 1])
   #clf = LinearDiscriminantAnalysis(n_components=1).fit_tranform(X,Y)
    clf = LinearDiscriminantAnalysis()
    clf.fit(X, y)
    print(clf.predict([[-0.8, -1]]))
if __name__=="__main__":
    main()
