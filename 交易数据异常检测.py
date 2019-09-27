import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

##########################################################################################################################################



data = pd.read_csv("creditcard.csv")

#print(data.head())
'''''
count_classes = pd.value_counts(data['Class'], sort=True).sort_index()
count_classes.plot(kind='bar')
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")
#plt.show()
'''''
data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
data = data.drop(['Time', 'Amount'], axis=1)
#print(data.head())

#下采样
X = data.iloc[:, data.columns != 'Class']
y = data.iloc[:, data.columns == 'Class']

number_records_fraud = len(data[data.Class == 1])
fraud_indices = np.array(data[data.Class == 1].index)

normal_indices = data[data.Class == 0].index

random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)#随机选择,在normal_indices中选择number_records_fraud个数
random_normal_indices = np.array(random_normal_indices)

under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])#连接

under_sample_data = data.iloc[under_sample_indices, :]#定位

X_undersample = under_sample_data.iloc[:, under_sample_data.columns !='Class']
y_undersample = under_sample_data.iloc[:, under_sample_data.columns == 'Class']

print("Percentage of normal transactions:", len(under_sample_data[under_sample_data.Class == 0])/len(under_sample_data))
print("Percentage of fraud transactions：", len(under_sample_data[under_sample_data.Class == 1])/len(under_sample_data))
print("Total number of transactions in resampled data:", len(under_sample_data))
#整个数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)#random_state=0每次随机得到的数据集一样

print("Number transactions train dataset:", len(X_train))
print("Number transaction test dataset:", len(X_test))
print("Total number of transactions:", len(X_train)+len(X_test))
#下采样数据集
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample, y_undersample, test_size=0.3, random_state=0)

print("")
print("Number transactions train dataset:", len(X_train_undersample))
print("Number transaction test dataset:", len(X_test_undersample))
print("Total number of transactions:", len(X_train_undersample)+len(X_test_undersample))




#Recall = TP/（TP+FN)
from sklearn.linear_model import LogisticRegression
#from sklearn.cross_validation import KFold, cross_val_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, recall_score, classification_report

def printing_Kfold_scores(x_train_data, y_train_data):
    #fold = KFold(len(y_train_data), 5, shuffle=False)
    fold1 = KFold(5, shuffle=False)
    fold1.get_n_splits(x_train_data)
    fold = fold1.split(x_train_data)

    #Different C parameters
    c_param_range = [0.01, 0.1, 1, 10, 100]

    results_table = pd.DataFrame(index=range(len(c_param_range), 2), columns=['C_parameter', 'Mean recall'])
    results_table['C_parameter'] = c_param_range

    j = 0
    for c_param in c_param_range:
        print('--------------------------------------------------------------')
        print('C parameter:', c_param)
        recall_accs = []
        for iteration, indices in enumerate(fold, start=1):

            lr = LogisticRegression(C=c_param, penalty='l1', solver='liblinear')

            lr.fit(x_train_data.iloc[indices[0], :], y_train_data.iloc[indices[0], :].values.ravel())

            y_pred_undersample = lr.predict(x_train_data.iloc[indices[1], :].values)

            recall_acc = recall_score(y_train_data.iloc[indices[1], :].values, y_pred_undersample)
            recall_accs.append(recall_acc)
            print('Iteration', iteration, ': recall score =', recall_acc)

        results_table.loc[j, 'Mean recall score'] = np.mean(recall_accs)
        j += 1
        print('')
        print('Mean recall score', np.mean(recall_accs))
        print('')

    best_c = results_table.loc[results_table['Mean recall score'].idxmax()]['C_parameter']
    print('*********************************************************************************')
    print('Best model to choose from cross validation is with C parameter = ', best_c)
    print('*********************************************************************************')

    return best_c


#best_c = printing_Kfold_scores(X_train, y_train)

if __name__ == '__main__':
    best_c = printing_Kfold_scores(X_train_undersample, y_train_undersample)
    # 1
    #best_c = printing_Kfold_scores(os_features, os_labels)
    #best_c = printing_Kfold_scores(X_train, y_train)

'''''
    # 过采样
    import pandas as pd
    from imblearn.over_sampling import SMOTE
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split
    credit_cards = pd.read_csv('creditcard.csv')
    columns = credit_cards.columns
    features_columns = columns.delete(len(columns) - 1)

    features = credit_cards[features_columns]
    labels = credit_cards['Class']

    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2,
                                                                                random_state=0)

    oversampler = SMOTE(random_state=0)
    os_features, os_labels = oversampler.fit_sample(features_train, labels_train)

    len(os_labels[os_labels == 1])
    os_features = pd.DataFrame(os_features)
    os_labels = pd.DataFrame(os_labels)
    best_c = printing_Kfold_scores[os_features, os_labels]
'''''
'''''
    lr = LogisticRegression(C=best_c, penalty='l1')
    lr.fit(X_train, y_train.values.ravel())
    y_pred_undersample = lr.predict(X_test.values)

    cnf_matrix = confusion_matrix(y_test, y_pred_undersample)
    np.set_printoptions(precision=2)

    print("Recall metric in the testing dataset:", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))

    class_names = [0, 1]
    #fig1 = plt.figure()
    #plot_confusion_matrix(confusion_matrix, classes=class_names, title='Confusion matrix')

    plt.show()

''''''

''
#2
lr = LogisticRegression(C=0.01, penalty='l1')
lr.fit(X_train_undersample, y_train_undersample.values.ravel())
y_pred_undersample_proba = lr.predict_proba(X_test_undersample.values)

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

plt.figure(figsize=(10, 10))

j = 1

for i in thresholds:
    y_test_predictions_high_recall = y_pred_undersample_proba[:, 1] > i

    plt.subplot(3, 3, j)
    j += 1


    cnf_matrix = confusion_matrix(y_test_undersample, y_test_predictions_high_recall)
    np.set_printoptions(precision=2)

    print("Recall metric in the testing dataset:", cnf_matrix[1, 1]/(cnf_matrix[1, 0]+cnf_matrix[1, 1]))

    class_names =[0, 1]
    #plot_confusion_matrix(cnf_matrix, classes=class_names, title='Threshold >=.show()
'''


'''''
lr = LogisticRegression(C=best_c, penalty='l1')
lr.fit(os_features, os_labels.values.ravel())
y_pred = lr.predict(features_test.values)

cnf_matrix = confusion_matrix(labels_test, y_pred)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset:", cnf_matrix[1, 1]/(cnf_matrix[1, 0]+cnf_matrix[1, 1]))

class_names = [0, 1]
plt.figure()
#plot_confusin_matrix()

plt.show()
'''''