import pandas as pd
import numpy as np

def main():
    import scipy.stats as ss
    print('正态检验',ss.normaltest(ss.norm.rvs(size=10)))#正态检验
    print('卡四方表格',ss.chi2_contingency([[15, 95], [85, 5]], False))#卡方四格表
    print('独立分布检验',ss.ttest_ind(ss.norm.rvs(size=10), ss.norm.rvs(size=20)))#t独立分布检验
    print('F分布检验',ss.f_oneway([49, 50, 39,40,43], [28, 32, 30,26,34], [38,40,45,42,48]))#F分布检验
    from statsmodels.graphics.api import qqplot
    from matplotlib import pyplot as plt
    qqplot(ss.norm.rvs(size=100))#QQ图
    plt.show()

    s = pd.Series([0.1, 0.2, 1.1, 2.4, 1.3, 0.3, 0.5])
    df = pd.DataFrame([[0.1, 0.2, 1.1, 2.4, 1.3, 0.3, 0.5], [0.5, 0.4, 1.2, 2.5, 1.1, 0.7, 0.1]]) #s1,s2
    
df = pd.DataFrame(np.array([s1,s2]).T)
    #相关分析
    print(s.corr(pd.Series([0.5, 0.4, 1.2, 2.5, 1.1, 0.7, 0.1])))

    print(df.corr())
    print(df.corr(method="spearman"))  #斯皮尔曼系数

    import numpy as np
    #回归分析
    x = np.arange(10).astype(np.float).reshape((10, 1))
    y = x * 3 + 4 + np.random.random((10, 1))
    print(x)
    print(y)
    from sklearn.linear_model import LinearRegression
    linear_reg = LinearRegression()#构建回归
    reg = linear_reg.fit(x, y)#拟合
    y_pred = reg.predict(x)#估计值
    print(reg.coef_)#参数
    print(reg.intercept_)#截距
    print(y.reshape(1, 10))
    print(y_pred.reshape(1, 10))
    plt.figure()
    plt.plot(x.reshape(1, 10)[0], y.reshape(1, 10)[0], "r*")
    plt.plot(x.reshape(1, 10)[0], y_pred.reshape(1, 10)[0])
    plt.show()

    #PCA降维（用的是奇异值分解的方法）
    df = pd.DataFrame(np.array([np.array([2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2, 1, 1.5, 1.1]),
                                np.array([2.4, 0.7, 2.9, 2.2, 3, 2.7, 1.6, 1.1, 1.6, 0.9])]).T)
    from sklearn.decomposition import PCA
    lower_dim = PCA(n_components=1)#降为1维
    lower_dim.fit(df.values)#转化后的
    print("PCA")
    print(lower_dim.explained_variance_ratio_)#维度的重要性
    print(lower_dim.explained_variance_)#转化后的数值

from scipy import linalg
#一般线性PCA函数
def pca(data_mat, topNfeat=1000000):#如果没有这么大的维度就全取
    mean_vals = np.mean(data_mat, axis=0)#每个属性的均值
    mid_mat = data_mat - mean_vals
    cov_mat = np.cov(mid_mat, rowvar=False)#斜方差，针对列
    eig_vals, eig_vects = linalg.eig(np.mat(cov_mat))#求斜方差矩阵的特征值和特征项量
    eig_val_index = np.argsort(eig_vals)#取出最大的特征值对应的特征向量，得到排序后的下标
    eig_val_index = eig_val_index[:-(topNfeat + 1):-1]#
    eig_vects = eig_vects[:, eig_val_index]#取出特征向量
    low_dim_mat = np.dot(mid_mat, eig_vects)#转换
    # ret_mat = np.dot(low_dim_mat,eig_vects.T)
    return low_dim_mat, eig_vals#转化后的矩阵，特征值


if __name__=="__main__":
    main()
