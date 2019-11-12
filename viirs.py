import pandas as pd
import numpy as np
import sklearn.cluster
from sklearn.cluster import KMeans
viirs=pd.read_csv("C:/Users/lenovo/Desktop/viirsdmps1.csv",encoding='gb2312')
print(viirs.head(3))
import matplotlib.pyplot as plt

viirs['logviirs']=np.log(viirs['DNviirs'])
viirs['logdmsp']=np.log(viirs['DNdmsp'])
data=viirs[['DNviirs','logdmsp']]
#假如我要构造一个聚类数为3的聚类器
estimator = KMeans(n_clusters=3)#构造聚类器
estimator.fit(data)#聚类
label_pred = estimator.labels_ #获取聚类标签
centroids = estimator.cluster_centers_ #获取聚类中心
inertia = estimator.inertia_ # 获取聚类准则的总和
clf = KMeans(n_clusters=10)
X1=data[['logdmsp','DNviirs']]

#画出聚类结果，每一类用一种颜色
SSE = []  # 存放每次结果的误差平方和
for k in range(1,9):
    estimator = KMeans(n_clusters=k)  # 构造聚类器
    estimator.fit(X1)
    SSE.append(estimator.inertia_)
Y = range(1,9)
#plt.xlabel('k')
#plt.ylabel('SSE')
#plt.plot(Y,SSE,'o-')
#plt.show()
from sklearn.metrics import silhouette_score
Scores = []  # 存放轮廓系数
for k in range(2,9):
    estimator = KMeans(n_clusters=k)  # 构造聚类器
    estimator.fit(X1)
    Scores.append(silhouette_score(X1,estimator.labels_,metric='euclidean'))
Y = range(2,9)
plt.xlabel('k')
plt.ylabel('轮廓系数')
#plt.plot(Y,Scores,'o-')
#plt.show()
#选轮廓系数大，sse小的

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
plt.subplot(221)
plt.scatter(viirs['logdmsp'],viirs['DNviirs'],s=4)
plt.xlabel('logDNdmsp')
plt.ylabel('DNviirs')
plt.subplot(222)
plt.scatter(viirs['logdmsp'],viirs['logviirs'],s=4)
plt.xlabel('logDNdmsp')
plt.ylabel('logDNviirs')
plt.subplot(223)
plt.scatter(viirs['DNdmsp'],viirs['logviirs'],s=4)
plt.xlabel('DNdmsp')
plt.ylabel('logDNviirs')
plt.subplot(224)
plt.scatter(viirs['DNdmsp'],viirs['DNviirs'],s=4)
plt.xlabel('DNdmsp')
plt.ylabel('DNviirs')
fig.subplots_adjust(left=0.15, top=0.95)
#plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


# 直线方程函数
def f_1(x, A, B,C):
    return A * np.exp(B*x) + C


# 二次曲线方程
def f_2(x, A, B, C):
    return A * x * x + B * x + C


# 三次曲线方程
def f_3(x, A, B,C):
    return A *x**B+C


def plot_test():
    plt.figure()
    # 拟合点
    x0 = viirs['logdmsp']
    y0 = viirs['logviirs']

    # 绘制散点
    plt.scatter(x0[:], y0[:], 4, "red")

    # 直线拟合与绘制
    A1, B1 ,C1= optimize.curve_fit(f_3, x0, y0)[0]
    x1 = x0
    y1 =  A1 * x1**B1 + C1
    plt.plot(x1, y1, "blue")
    print(A1,B1,C1)
    x2=x0
    y2=  A1 * np.exp(B1*x2) + C1
    #plt.plot(x2, y2, "green")

    #from astropy.units import Ybarn
    import math

    def computeCorrelation(X, Y):
        xBar = np.mean(X)
        yBar = np.mean(Y)
        SSR = 0
        varX = 0
        varY = 0
        for i in range(0, len(X)):
            diffXXBar = X[i] - xBar
            diffYYBar = Y[i] - yBar
            SSR += (diffXXBar * diffYYBar)
            varX += diffXXBar ** 2
            varY += diffYYBar ** 2

        SST = math.sqrt(varY * varY)
        return SSR / SST

    testX = y2
    testY = y0

    print(computeCorrelation(testX, testY))

    # 二次曲线拟合与绘制
    #A2, B2, C2 = optimize.curve_fit(f_1, x0, y0)[0]
    #x2 = np.arange(-6,4 , 0.01)
    #y2 = A2 * x2 * x2 + B2 * x2 + C2
    #plt.plot(x2, y2, "green")

    #A3, B3= optimize.curve_fit(f_3, x0, y0)[0]
   # x3 = np.arange(-6,4 , 0.01)
    #y3 = A3 * B3**x3
    #plt.plot(x3, y3, "purple")

    plt.title("test")
    plt.xlabel('logdmsp')
    plt.ylabel('viirs')

    plt.show()

    return
print(plot_test())
