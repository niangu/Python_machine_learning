#信用卡欺诈检测
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("creditcard.csv")
print(data.head())

count_classes = pd.value_counts(data['Class'], sort=True).sort_index()
print(count_classes)
#value_counts统计Class列的属性值0，1各有多少个
count_classes.plot(kind='bar') #条形图，可以用pandas直接画
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.xticks(rotation=0)
plt.show()

from sklearn.preprocessing import StandardScaler
data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1)) #转换成新的特征
data = data.drop(['Time', 'Amount'], axis=1)#删除没用的特征
print(data.head())

X = data.ix[:, data.columns != 'Class']  #拿出所有样本 ，不包括Class列
y = data.ix[:, data.columns == 'Class']  ##拿出所有样本 ，只拿出Class列，我们的label列

# Number of data points in the minority class
number_records_fraud = len(data[data.Class == 1]) #class=1的有多少个
fraud_indices = np.array(data[data.Class == 1].index) #把class=1的样本的索引拿出来
#print(fraud_indices)
# Picking the indices of the normal classes
normal_indices = data[data.Class == 0].index  #把class=0的样本的索引拿出来

# Out of the indices we picked, randomly select "x" number (number_records_fraud) replace是否进行代替
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
random_normal_indices = np.array(random_normal_indices)
#拿出索引后再转换为np.array格式

# Appending the 2 indices 合并索引
under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])

# Under sample dataset  现在合并之后的数据
under_sample_data = data.iloc[under_sample_indices, :]

X_undersample = under_sample_data.ix[:, under_sample_data.columns != 'Class']
y_undersample = under_sample_data.ix[:, under_sample_data.columns == 'Class']

# Showing ratio
print("Percentage of normal transactions: ", len(under_sample_data[under_sample_data.Class == 0])/len(under_sample_data))
print("Percentage of fraud transactions: ", len(under_sample_data[under_sample_data.Class == 1])/len(under_sample_data))
print("Total number of transactions in resampled data: ", len(under_sample_data))

from sklearn.model_selection import train_test_split
# Whole dataset 原始数据集  将来用原始的test集测试
# 交叉验证 先进行切分 随机切分 test_size切分比例  random_state=0随机
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("Number transactions train dataset: ", len(X_train))
print("Number transactions test dataset: ", len(X_test))
print("Total number of transactions: ", len(X_train)+len(X_test))

# Undersampled dataset下采样的数据集
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample, y_undersample, test_size=0.3, random_state=0)
print("")
print("Number transactions train dataset: ", len(X_train_undersample))
print("Number transactions test dataset: ", len(X_test_undersample))
print("Total number of transactions: ", len(X_train_undersample)+len(X_test_undersample))

# 召回率Recall = TP/(TP+FN) ,异常值检测问题用recall值来评估模型效果
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold,cross_val_score
from sklearn.metrics import confusion_matrix,recall_score,classification_report


# 定义一个返回交叉验证分数的函数
def printing_KFold_scores(x_train_data, y_train_data, c_param):
    fold = KFold(n_splits=5, shuffle=False)
    fold.get_n_splits(x_train_data)

    results_table = pd.DataFrame(index=range(5, 2), columns=['C_parameter', 'Mean recall score'])
    results_table['Mean recall score'] = results_table['Mean recall score'].astype('float64')

    # kfold 会给出两个列表：train_indices = indices[0], test_indices = indices[1]
    j = 0

    recall_accs = []
    for iteration, indices in enumerate(fold.split(x_train_data), start=1):  # 交叉验证

    # 具体化模型
        lr = LogisticRegression(C=c_param, penalty='l1')

    # 训练模型
    lr.fit(x_train_data.iloc[indices[0], :], y_train_data.iloc[indices[0], :].values.ravel())

    # 模型预测
    y_pred_undersample = lr.predict(x_train_data.iloc[indices[1], :].values)

    # 计算召回分数
    recall_acc = recall_score(y_train_data.iloc[indices[1], :].values, y_pred_undersample)
    recall_accs.append(recall_acc)
    print('Iteration', iteration, ':recall score = ', recall_acc)

    # 计算每个c值对应的平均召回分数
    results_table.ix[j, 'Mean recall score'] = np.mean(recall_accs)
    j += 1
    print('')
    print('Mean recall score:', np.mean(recall_accs))
    print('')

    best_c = printing_KFold_scores(X_train_undersample,y_train_undersample,0.01)

#best_c = printing_KFold_scores(X_train_undersample, y_train_undersample, 0.1)

#best_c = printing_KFold_scores(X_train_undersample,y_train_undersample,1)

#best_c = printing_KFold_scores(X_train_undersample,y_train_undersample,10)

#best_c = printing_KFold_scores(X_train_undersample,y_train_undersample,100)

# 定义一个混淆矩阵作图的函数

import itertools
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


lr = LogisticRegression(C = 0.01, penalty = 'l1')
lr.fit(X_train_undersample,y_train_undersample.values.ravel())
y_pred_undersample = lr.predict(X_test_undersample.values)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test_undersample,y_pred_undersample)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()

# 对原测试数据集进行预测，发现误杀率太高，这是由于下采样的原因
lr = LogisticRegression(C = 0.01, penalty = 'l1')
lr.fit(X_train_undersample,y_train_undersample.values.ravel())
y_pred = lr.predict(X_test.values)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test,y_pred)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()

lr = LogisticRegression(C=0.01, penalty='l1')
lr.fit(X_train_undersample, y_train_undersample.values.ravel())
y_pred_undersample_proba = lr.predict_proba(X_test_undersample.values)
# 将阈值分别设置为下列数据，分析每种情况下的混淆矩阵，得到最优阈值
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

plt.figure(figsize=(10, 10))

j = 1
for i in thresholds:
    y_test_predictions_high_recall = y_pred_undersample_proba[:, 1] > i

    plt.subplot(3, 3, j)
    j += 1

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test_undersample, y_test_predictions_high_recall)
    np.set_printoptions(precision=2)

    print("Recall metric in the testing dataset: ", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))

    # Plot non-normalized confusion matrix
    class_names = [0, 1]
    plot_confusion_matrix(cnf_matrix
                          , classes=class_names
                          , title='Threshold >= %s' % i)
plt.show()



'''''
###########################################################################################################################################################
#过采样
# 导入相关库
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

credit_cards = pd.read_csv('creditcard.csv')

columns = credit_cards.columns
# 特征列名
features_columns = columns.delete(len(columns)-1)
# 特征值
features = credit_cards[features_columns]
# label值
labels = credit_cards['Class']


# 划分训练集合测试集
features_train,features_test,labels_train,labels_test = train_test_split(features,
                                                                        labels,
                                                                        test_size= 0.2,
                                                                        random_state = 0)

# 过采样
oversampler = SMOTE(random_state=0)
os_features,os_labels = oversampler.fit_sample(features_train,labels_train)
os_features = pd.DataFrame(os_features)
os_labels = pd.DataFrame(os_labels)

print(len(os_labels[os_labels==1]))


lr = LogisticRegression(C = 0.01, penalty = 'l1')
lr.fit(os_features,os_labels.values.ravel())
y_pred = lr.predict(features_test.values)

# Compute confusion matrix
cnf_matrix = confusion_matrix(labels_test,y_pred)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()
'''''