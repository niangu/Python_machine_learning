import xgboost as xgb
import pandas as pd
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings('ignore')

#数据预处理
train = pd.read_csv('data/train.csv')
#对数转换
train['log_loss'] = np.log(train['loss'])
#数据分成连续和离散特征
features = [x for x in train.columns if x not in ['id', 'loss', 'log_loss']]
cat_features = [x for x in train.select_dtypes(
    include=['object']).columns if x not in ['id', 'loss', 'log_loss']]
num_features = [x for x in train.select_dtypes(
    exclude=['object']).columns if x not in ['id', 'loss', 'log_loss']]

print("Categorical features:", len(cat_features))
print("Numerical features:", len(num_features))

ntrain = train.shape[0]

train_x = train[features]
train_y = train['log_loss']

for c in range(len(cat_features)):
    train_x[cat_features[c]] = train_x[cat_features[c]].astype('category').cat.codes

print("Xtrain:", train_x.shape)
print('ytrain:', train_y.shape)

#Simple XGBoost Model

def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y), np.exp(yhat))

#Model
dtrain = xgb.DMatrix(train_x, train['log_loss'])

''''
Xgboost参数
'booster':'gbtree',
'objective': 'multi:softmax', 多分类的问题
'num_class':10, 类别数，与 multisoftmax 并用
'gamma':损失下降多少才进行分裂
'max_depth':12, 构建树的深度，越大越容易过拟合
'lambda':2, 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
'subsample':0.7, 随机采样训练样本
'colsample_bytree':0.7, 生成树时进行的列采样
'min_child_weight':3, 孩子节点中最小的样本权重和。如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束
'silent':0 ,设置成1则没有运行信息输出，最好是设置为0.
'eta': 0.007, 如同学习率
'seed':1000,
'nthread':7, cpu 线程数
xgb_params = {
    'seed': 0,
    'eta': 0.1,
    'colsample_bytree': 0.5,
    'silent': 1,
    'subsample': 0.5,
    'objective': 'reg:linear',
    'max_depth': 5,
    'min_child_weight': 3
}

'''

xgb_params = {
    'seed': 0,
    'eta': 0.1,
    'colsample_bytree': 0.5,
    'silent': 1,
    'subsample': 0.5,
    'objective': 'reg:linear',
    'max_depth': 5,
    'min_child_weight':3
}
#使用交叉验证xgb.cv
bst_cv1 = xgb.cv(xgb_params, dtrain, num_boost_round=50, nfold=3, seed=0, feval=xg_eval_mae, maximize=False, early_stopping_rounds=10)
print('CV score:', bst_cv1.iloc[-1, :]['test-mae-mean'])
plt.figure()
bst_cv1[['train-mae-mean', 'test-mae-mean']].plot()
plt.show()

#第一个基础模型
#建立100个树模型
bst_cv2 = xgb.cv(xgb_params, dtrain, num_boost_round=100, nfold=3, seed=0, feval=xg_eval_mae, maximize=False, early_stopping_rounds=10)
print('CV score:', bst_cv2.iloc[-1, :]['test-mae-mean'])

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(16, 4)

ax1.set_title('100 rounds of training')
ax1.set_xlabel('Rounds')
ax1.set_ylabel('Loss')
ax1.grid(True)
ax1.plot(bst_cv2[['train-mae-mean', 'test-mae-mean']])
ax1.legend(['Training Loss', 'Test Loss'])

ax2.set_title('60 last rounds of training')
ax2.set_xlabel('Rounds')
ax2.set_ylabel('Loss')
ax2.grid(True)
ax2.plot(bst_cv2.iloc[40:][['train-mae-mean', 'test-mae-mean']])
ax2.legend(['Training Loss', "Test Loss"])
plt.show()

#XGBoost参数调节
#Step1:选择一组初始参数
#Step2:改变max_depth和min_child_weight
#Step3:调节gamma降低过拟合风险
#Step3:调节subsample 和colsample_bytree改变数据采样策略
#Step5: 调节学习率eta
class XGBoostRegressor(object):
    def __init__(self, **kwargs):
        self.params = kwargs
        if 'num_boost_round' in self.params:
            self.num_boost_round = self.params['num_boost_round']
        self.params.update({'silent': 1, 'objective': 'reg:linear', 'seed': 0})
    def fit(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, y_train)
        self.bst = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=self.num_boost_round,
                            feval=xg_eval_mae, maximize=False)
    def predict(self, x_pred):
        dpred = xgb.DMatrix(x_pred)
        return self.bst.predict(dpred)
    def kfold(self, x_train, y_train, nfold=5):
        dtrain = xgb.DMatrix(x_train, y_train)
        cv_rounds = xgb.cv(params=self.params, dtrain=dtrain, num_boost_round=self.num_boost_round, nfold=nfold, feval=xg_eval_mae, maximize=False, early_stopping_rounds=10)
        return cv_rounds.iloc[-1, :]
    def plot_feature_importances(self):
        feat_imp = pd.Series(self.bst.get_fscore()).sort_values(ascending=False)
        feat_imp.plot(title='Feature Importances')
        plt.ylabel('Feature Importance Score')
    def get_params(self, deep=True):
        return self.params
    def set_params(self, **params):
        self.params.update(params)
        return self


def mae_score(y_true, y_pred):
    return mean_absolute_error(np.exp(y_true), np.exp(y_pred))

mae_scorer = make_scorer(mae_score, greater_is_better=False)
bst = XGBoostRegressor(eta=0.1, colsample_bytree=0.5, subsample=0.5,
                       max_depth=5, min_child_weight=3, num_boost_round=50)
bst.kfold(train_x, train_y, nfold=5)
''''
Step 1: 学习率与树个数
Step 2: 的深度与节点权重
这些参数对xgboost性能影响最大，因此，他们应该调整第一。我们简要地概述它们：

max_depth: 树的最大深度。增加这个值会使模型更加复杂，也容易出现过拟合，深度3-10是合理的。

min_child_weight: 正则化参数. 如果树分区中的实例权重小于定义的总和，则停止树构建过程。
'''
xgb_param_grid = {'max_depth':list(range(4, 9)), 'min_child_weight':list((1, 3, 6))}
xgb_param_grid['max_depth']

grid = GridSearchCV(XGBoostRegressor(eta=0.1, num_boost_round=50, colsample_bytree=0.5, subsample=0.5), param_grid=xgb_param_grid, cv=5, scoring=mae_scorer)
grid.fit(train_x, train_y.values)
grid.grid_scores_, grid.best_params_, grid.best_score_


def convert_grid_scores(scores):
    _params = []
    _params_mae = []
    for i in scores:
        _params.append(i[0].values())
        _params_mae.append(i[1])
    params = np.array(_params)
    grid_res = np.column_stack((_params, _params_mae))
    return [grid_res[:, i] for i in range(grid_res.shape[1])]

_,scores = convert_grid_scores(grid.grid_scores_)
scores = scores.reshape(5, 3)

plt.figure(figsize=(10, 5))
cp = plt.contourf(xgb_param_grid['min_child_weight'], xgb_param_grid['max_depth'], scores, cmap='BrBG')
plt.colorbar(cp)
plt.title('Depth / min_child_weight optimization')
plt.annotate('We use this', xy=(5.95, 7.95), xytext=(4, 7.5), arrowpeops=dict(facecolor='white'), color='white')
plt.annotate('Good for depth=7', xy=(5.98, 7.05), xytext=(4, 6.5), arrowprops=dict(facecolor='white'), color='white')
plt.xlabel('min_child_weight')
plt.ylabel('max_depth')
plt.grid(True)
plt.show()

#Step3:调节gamma去降低过拟合风险
xgb_param_grid = {'gamma':[0.1 * i for i in range(0, 5)]}
grid = GridSearchCV(XGBoostRegressor(eta=0.1, num_boost_round=50, max_depth=8, min_child_weight=6, colsample_bytree=0.5, subsample=0.5),
                    param_grid=xgb_param_grid, cv=5, scoring=mae_scorer)
grid.fit(train_x, train_y.values)
print(grid.grid_scores_, grid.best_params_, grid.best_score_)

#Step4：调节样本采样方式subsample 和colsample_bytree
xgb_param_grid = {'subsample':[0.1 * i for i in range(6,9)], 'colsample_bytree':[0.1 * i for i in range(6, 9)]}
grid = GridSearchCV(XGBoostRegressor(eta=0.1, gamma=0.2, num_boost_round=50, max_depth=8, min_child_weight=6),
                    param_grid=xgb_param_grid, cv=5, scoring=mae_scorer)
grid.fit(train_x, train_y.values)
print(grid.grid_scores_, grid.best_params_, grid.best_score_)

_, scores = convert_grid_scores(grid.grid_scores_)
scores = scores.reshape(3, 3)

plt.figure(figsize=(10, 5))

cp = plt.contourf(xgb_param_grid['subsample'], xgb_param_grid['colsample_bytree'], scores, cmap='BrBG')
plt.colorbba(cp)
plt.title('Subsampling params tuning')
plt.annotate('Optimum', xy=(0.895, 0.6), xytext=(0.8, 0.695), arrowprops=dict(facecolor='black'))
plt.xlabel('subsample')
plt.ylabel('colsample_bytree')
plt.grid(True)
plt.show()


#Step5:减少学习率并增大树的个数
xgb_param_grid = {'eta':[0.5, 0.4, 0.3, 0.2, 0.1, 0, 0.075, 0.05, 0.04, 0.03]}
grid = GridSearchCV(XGBoostRegressor(num_boost_round=50, gamma=0.2, max_depth=8, min_child_weight=6, colsample_bytree=0.6, subsample=0.9),
                    param_grid=xgb_param_grid, cv=5, scoring=mae_scorer)
grid.fit(train_x, train_y.values)
print(grid.grid_scores_, grid.best_params_, grid.best_score_)

eta, y = convert_grid_scores(grid.grid_scores_)
plt.figure(figsize=(10, 4))
plt.title('MAE and ETA, 50 tress')
plt.xlabel('eta')
plt.ylabel('score')
plt.plot(eta, -y)
plt.grid(True)
plt.show()

xgb_param_grid = {'eta':[0.5, 0.4, 0.3, 0.2, 0.1, 0.075, 0.05, 0.04, 0.03]}
grid = GridSearchCV(XGBoostRegressor(num_boost_round=100, gamma=0.2, max_depth=8, min_child_weight=6,
                                     colsample_bytree=0.6, subsample=0.9),
                    param_grid=xgb_param_grid, cv=5, scoring=mae_scorer)
grid.fit(train_x, train_y.values)
print(grid.grid_scores, grid.best_params_, grid.best_score_)

eta, y = convert_grid_scores(grid.grid_scores_)
plt.figure(figsize=(10, 4))
plt.xlabel('eta')
plt.ylabel('score')
plt.plot(eta, -y)
plt.grid(True)
plt.show()

xgb_param_grid = {'eta': [0.09, 0.08, 0.07, 0.06, 0.05, 0.04]}
grid = GridSearchCV(XGBoostRegressor(num_boost_round=200, gamma=0.2, max_depth=8, min_child_weight=6,
                                     colsample_bytree=0.6, subsample=0.9),
                    param_grid=xgb_param_grid, cv=5, scoring=mae_scorer)
grid.fit(train_x, train_y.values)
print(grid.grid_scores_, grid.best_params_, grid.best_score_)

eta, y =convert_grid_scores(grid.grid_scores_)
plt.figure(figsize=(10, 4))
plt.title('MAE and ETA, 200 trees')
plt.xlabel('eta')
plt.ylabel('score')
plt.plot(eta, -y)
plt.grid(True)
plt.show()

#Final XGBoost model

bst = XGBoostRegressor(num_boost_round=200, eta=0.07, gamma=0.2, max_depth=8, min_child_weight=6,
                       colsample_bytree=0.6, subsample=0.9)
cv = bst.kfold(train_x, train_y, nfold=5)
print(cv)