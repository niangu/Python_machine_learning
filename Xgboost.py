import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import cross_val_score

from scipy import stats
import seaborn as sns
from copy import deepcopy

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
print(train.shape)

print('First 20 columns:', list(train.columns[:20]))
print('Last 20 columns:', list(train.columns[-20:]))
print(train.describe())
print(pd.isnull(train).values.any())

#Continuous vs caterogical features
print(train.info())
cat_features = list(train.select_dtypes(include=['object']).columns)
print("Categorical: {} features".format(len(cat_features)))
cont_features = [cont for cont in list(train.select_dtypes(include=['float64', 'int64']).columns) if cont not in ['loss', 'id']]
print("Continuous: {} features".format(len(cont_features)))

id_col = list(train.select_dtypes(include=['int64']).columns)
print("A column of int64: {}".format(id_col))
#类别值中属性的个数
cat_uniques = []
for cat in cat_features:
    cat_uniques.append(len(train[cat].unique()))

uniq_values_in_categories = pd.DataFrame.from_items([('cat_name', cat_features), ('unique_values', cat_uniques)])
print(uniq_values_in_categories.head())

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(16, 5)
ax1.hist(uniq_values_in_categories.unique_values, bins=50)
ax1.set_title('Amount of categorical features with X distinct values')
ax1.set_xlabel('Distinct values in a feature')
ax1.set_ylabel('Features')
ax1.annotate('A feature with 326 vals', xy=(322, 2), xytext=(200, 38), arrowprops=dict(facecolor='black'))

ax2.set_xlim(2, 30)
ax2.set_title('Zooming in the [0, 30] part of left histogram')
ax2.set_xlabel("Distinct values in a feature")
ax2.set_ylabel('Features')
ax2.grid(True)
ax2.hist(uniq_values_in_categories[uniq_values_in_categories.unique_values <= 30].unique_values, bins=30)
ax2.annotate('Binary features', xy=(3, 71), xytext=(7, 71), arrowprops=dict(facecolor='black'))
plt.show()

#赔偿值
plt.figure(figsize=(16, 8))
plt.plot(train['id'], train['loss'])
plt.title('Loss values per id')
plt.xlabel('id')
plt.ylabel('loss')
plt.legend()
plt.show()

stats.mstats.skew(train['loss']).data
stats.mstats.skew(np.log(train['loss'])).data

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(16, 5)
ax1.hist(train['loss'], bins=50)
ax1.set_title('Train Loss target histogram')
ax1.grid(True)
ax2.hist(np.log(train['loss']), bins=50, color='g')
ax2.set_title('Train Log Loss Target histogram')
ax2.grid(True)
plt.show()

#连续值特征
train[cont_features].hist(bins=50, figsize=(16, 12))

#特征之间的 相关性
plt.subplots(figsize=(16, 9))
correlation_mat = train[cont_features].corr()
sns.heatmap(correlation_mat, annot=True)
plt.show()