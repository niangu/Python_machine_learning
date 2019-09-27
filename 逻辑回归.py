import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random
import time
from sklearn import preprocessing as pp
import os

STOP_ITER = 0
STOP_COST = 1
STOP_GRAD = 2
n = 100
'''''
The logistic regression
目标：建立分类器（求解三个参数)
设定阀值,根据阀值判断录取结果

要完成的模块：
sigmoid：映射到概率的函数
model:返回预测结果值
cost:根据参数计算损失
gradient:计算每个参数的梯度方向
descent:进行参数更新
accuracy:计算精度
'''''
def sigmoid(z):
       return 1/(1 + np.exp(-z))


def model(X, theta):
       return sigmoid(np.dot(X, theta.T))#np.dot矩阵乘法


def cost(X, y, theta):
       left = np.multiply(-y, np.log(model(X, theta)))
       right = np.multiply(1 - y, np.log(1 - model(X, theta)))
       print(np.sum(left - right) / (len(X)))
       return np.sum(left - right) / (len(X))


def gradient(X, y, theta):
       grad = np.zeros(theta.shape)
       error = (model(X, theta) - y).ravel()
       for j in range(len(theta.ravel())):
              term = np.multiply(error, X[:, j])
              grad[0, j] = np.sum(term) /len(X)

       return grad[0, j]


def stopCriterion(type, value, threshold):
       if type == STOP_ITER:
              return value > threshold
       elif type == STOP_COST:
              return abs(value[-1]-value[-2]) < threshold
       elif type == STOP_GRAD:
              return np.linalg.norm(value) < threshold


def shuffleData(data):
       np.random.shuffle(data)
       cols = data.shape[1]
       X = data[:, 0:cols-1]
       y = data[:, cols-1:]
       return X, y


def descent(data, theta, batchSize, stopType, thresh, alpha):
       init_time = time.time()
       i = 0
       k = 0
       X, y = shuffleData(data)
       grad = np.zeros(theta.shape)
       costs = [cost(X, y, theta)]
       while True:
              grad = gradient(X[k:k+batchSize], y[k:k+batchSize], theta)
              k += batchSize
              if k >= n:
                 k=0
                 X, y = shuffleData(data)
              theta = theta - alpha*grad
              costs.append(cost(X, y, theta))
              i += 1

              if stopType == STOP_ITER:
                 value = i
              elif stopType == STOP_COST:
                 value = costs
              elif stopType == STOP_GRAD:
                 value = grad
              if stopCriterion(stopType, value, thresh):
                 break
       return theta, i-1, costs, grad, time.time() - init_time


def runExpe(data, theta, batchSize, stopType, thresh, alpha):
       theta, iter, costs, grad, dur = descent(data, theta, batchSize, stopType, thresh, alpha)
       name = "Original" if (data[:, 1]>2).sum() > 1 else "Scaled"
       name += " data - learning rate: {} -".format(alpha)
       if batchSize == n:
              strDescType = "Gradient"
       elif batchSize ==1:
              strDescType = "Stochastic"
       else: strDescType = "Mini-batch({})".format(batchSize)
       name += strDescType + "descent - Stop:"
       if stopType == STOP_ITER:
              strStop = "{} iterations".format(thresh)
       elif stopType == STOP_COST:
              strStop = "costs change < {}".format(thresh)
       else:
              strStop = "gradient norm < {}".format(thresh)
       name += strStop
       print("***{}\nTheta: {} - Iter: {} -Last cost: {:03.2f} - Duration: {:03.2f}s".format(name, theta, iter, costs[-1], dur))
       fig, ax = plt.subplots(figsize=(12, 4))
       ax.plot(np.arange(len(costs)), costs, 'r')
       ax.set_xlabel('Iterations')
       ax.set_ylabel('Cost')
       ax.set_title(name.upper() + '- Error vs. Iteration')
       fig.show()
       return theta


def predict(X, theta):
       return [1 if x >= 0.5 else 0 for x in model(X, theta)]
def main():
       path = '/home/niangu/桌面/数据挖掘与建模/data' + os.sep + 'LogiReg_data.txt'
       pdData = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
       print(pdData.head())
       print(pdData.shape)
       positive = pdData[pdData['Admitted'] == 1]
       negative = pdData[pdData['Admitted'] == 0]
       fig, ax = plt.subplots(figsize=(10, 5))
       ax.scatter(positive['Exam 1'], positive['Exam 2'], s=30, c='b', marker='o', label='Admitted')
       ax.scatter(negative['Exam 1'], negative['Exam 2'], s=30, c='r', marker='x', label='Not Admitted')
       ax.legend()
       ax.set_xlabel('Exam 1 Score')
       ax.set_ylabel('Exam 2 Score')
       fig.show()

       nums = np.arange(-10, 10, step=1)
       fig, ax = plt.subplots(figsize=(12, 4))
       ax.plot(nums, sigmoid(nums), 'r')
       fig.show()
       ######
       pdData.insert(0, 'Ones', 1)

       orig_data = pdData.as_matrix()
       cols = orig_data.shape[1]
       X = orig_data[:, 0:cols-1]
       y = orig_data[:, cols-1:cols]
       theta = np.zeros([1, 3])
       print(theta)
       cost(X, y, theta)
       scaled_data = orig_data.copy()
       scaled_data[:, 1:3] = pp.scale(orig_data[:, 1:3])
       runExpe(scaled_data, theta, 16, STOP_GRAD, thresh=0.002*2, alpha=0.001)

       scaled_X = scaled_data[:, :3]
       y = scaled_data[:, 3]
       predictions = predict(scaled_X, theta)
       correct = [1 if (( a == 1 and b == 1) or ( a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
       accuracy = (sum(map(int, correct)) % len(correct))
       print('accuracy = {0}%'.format(accuracy))


if __name__ == '__main__':
       main()