import numpy

world_alcohol = numpy.genfromtxt("world_alcohol.txt", delimiter=",",dtype=str)
print(type(world_alcohol)
print(worls_alcohol)
print(help(nunmpy.genfromtxt))

vector = numpy.array([5, 10, 15, 20])
matrix = numpy.array([[5, 10, 15],[20, 25, 30], [35, 40, 45]]])
print vector
print matrix

vector = numpy.array([1, 2, 3 ,4])
print(vector.shape)#看一下结构
matrix = numpy.array([5, 10, 15], [20, 25, 30]])
print(matrix.shape)


worls_alcohol = numpy.genfromtxt("world_alcohol.txt, delimiter=",", dtype=str, skip_header=1)
uruguay_other_1986 = world_alcohol[1, 4] #取值
third_country = world_alchol[2,2]
print uruguay_other_1986
print third_country


vector = numpy.array([5, 10, 15, 20])#切片
print(vector[0:3])

matrix = numpy.array([
                    [5, 10, 15],
                    [20, 25, 30],
                    [35, 40, 45]
                    ])
print(matrix[:,1])#切片，切1列
print(matrix[:,0:2])#切2列
print(matrix[1:3,0:2])#行，列


vector = numpy.array([5, 10, 15, 20])
vector == 10 #相当于对每个都进行了判断
equal_to_ten = (vevtor == 10 )
print equal_to_ten
print vector[equal_to_ten])#bool索引返回值10

matrix = numpy.array([
                    [5, 10, 15],
                    [20, 25, 30],
                    [35, 40, 45]
                    ])

second_column_25 = (matrix[:,1] == 25)
print second_column_25
print (matrix[second_column_25, :])

vector = numpy.array([5, 10, 15, 20])
#& ,并且，并且, |， 或者，或者

vector = numpy.array(["1", "2", "3"])
print(vector.dtype)
print vector
vector = vector.astype(float)#类型转换
print vector.dtype
print vector
 
vector = numpy.arry([5, 10, 15])
vector.min()
print(help(numpy.array))


matrix = numpy.array([
                    [5, 10, 15],
                    [20, 25, 30],
                    [35, 40, 45]
                    ])
matrix.sum(axis=1)#每行求和，axis=0,按列求和

import numpy as np
print(np.arange(15))
a = np.arange(15).reshape(3,5)
a
a.shape#结构
a.ndim#几维
a.dtype.name#类型
a.size#几个元素

np.zeros((3,4))#初始化一个3行4列的矩阵
np.ones((2,3,4),dtype=np.int32)#初始值为1

np.arange(10, 30, 5) #生成一个10-30，间隔为5的
np.random.random((2,3))

from numpy import pi
np.linspace(0, 2*pi, 100)#0-2*pi,取100个数
np.sin(np.linspace(0, 2*pi, 100)) #可以对np.linspace作一系列操作


a= np.array([20,30,40,50])
b = np.arange(4)

c = a-b#对应值相减
b**2 #求平方
a<35


A = np.array( [[1,1],
              [0,1]])
B = np.array([[2,0],
              [3,4]])
print(A*B)
print(A.dot(B))#行和列相乘
print(np.dot(A, B))


import numpy as np 
B = np.arange(3)
print(B)
print(np.exp(B))#e的多少次幂
print(np.sqrt(B))

a= np.floor(10*np.random.random((3,4))#向下取整
print(a.ravel())#拉伸矩阵
a.shape = (6, 2)
print(a.T)#转质，行和列变换一下
a.reshape(3, -1)#-1表示默认值

a= np.floor(10*np.random(2,2))
b= np.floor(10*np.random(2,2))
print(np.hstack((a,b)))#横的拼
print(np.vstack((a,b)))#竖的拼
a= np.floor(10*np.random.random((2,12)))
np.hsplit(a,3))#横向分割为3个数组
np.vsplit(a,3))#纵向分割


a = np.arange(12)
b = a   #指向同一内存
print(b is a)
b.shape= 3,4
print(a.shape)
print(id(a))
print(id(b))


c = a.view()#指向不同的结构，但共用一个值
print(c is a)
c.shape = 2.6
print(a.shape)
c[0,4] = 1234
print(a)
print(id(a))
print(id(c))

d = a.copy()#深复制
d is a
d[0,0] = 9999
print(a)
print(b)



data = np.sin(np.arange(20)), reshape(5,4)
print(data)
ind = data.argmax(axis=0)
print(ind)
data_max = data[ind, range[data.shape[1])]
print(data_max)

a = np.arange(0, 40, 10)
print(a)
b = np.tile(a, (2,2))#行变成2倍，列变成2倍
print(b)


a = np.array([[4, 3, 5],[1, 2,1]])
print(a)
b = np.sort(a, axis=1)
a.sort(axis=1)
print(b)

a = np.array([4,3,1,2])
j = np.argsort(a)#得到默认从小到大排序的索引
print(j)
print(a[j])#输出从小到大的结果
















