#参考文献:https://github.com/sarvasvkulpati/LinearRegression/blob/master/LinearRegression.py
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np

X,y = make_regression(n_samples=100,n_features=1,noise=0.4,bias=50)
"""
生成100个样本,其中每个样本只有一个特征,以及100个标签。
print(X)
print(y)
"""

#工具函数。参数theta0,theta1
def plotLine(theta0, theta1, X, y):
    max_x = np.max(X) + 100
    min_x = np.min(X) - 100


    xplot = np.linspace(min_x, max_x, 1000)
    yplot = theta0 + theta1 * xplot



    plt.plot(xplot, yplot, color='#58b970', label='Regression Line')

    plt.scatter(X,y)
    plt.axis([-10, 10, 0, 200])
    plt.show()


#单层线性回归
def hypothesis(theta0,theta1,x):
    return theta0 + (theta1*x)
#损失函数
def cost(theta0,theta1,X,y):
    costValue = 0
    for (xi,yi) in  zip(X,y):
        costValue += 0.5 * ((hypothesis(theta0,theta1,xi) - yi)**2)
    return costValue
#损失函数-梯度计算
def derivatives(theta0,theta1,X,y):
    dtheta0 = 0
    dtheta1 = 0
    for (xi,yi) in zip(X,y):
        dtheta0 += hypothesis(theta0, theta1, xi) - yi
        dtheta1 += (hypothesis(theta0, theta1, xi) - yi) * xi
    dtheta0 /= len(X)
    dtheta1 /= len(X)
    return dtheta0,dtheta1
#更新参数w和b
def updateParameters(theta0, theta1, X, y, alpha):
    dtheta0,dtheta1 = derivatives(theta0,theta1,X,y)
    theta0 = theta0 - (alpha * dtheta0)
    heta1 = theta1 - (alpha * dtheta1)
    return theta0, theta1
#线性回归
def LinearRegression(X,y):
    #初始化参数
    theta0 = np.random.rand()
    theta1 = np.random.rand()
    #训练1000个批次
    for i in range(0,1000):
        if i % 100 == 0:
            plotLine(theta0, theta1, X, y)
        theta0, theta1 = updateParameters(theta0, theta1, X, y, 0.005)

LinearRegression(X, y)