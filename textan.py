
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.cluster
from sklearn.cluster import KMeans
viirs=pd.read_csv("C:/Users/lenovo/Desktop/viirsdmps1.csv",encoding='gb2312')
print(viirs.head(3))
print(viirs['DNviirs'].describe())
import matplotlib.pyplot as plt

viirs['logviirs']=np.log(viirs['DNviirs'])
viirs['logdmsp']=np.log(viirs['DNdmsp'])
data=viirs[['DNviirs','logdmsp']]
print(viirs['logviirs'].describe())
from sklearn.metrics import classification_report,accuracy_score,precision_score,mean_squared_error


import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


# 直线方程函数
def f_1(x, A, B):
    return A * x + B


# 二次曲线方程
def f_2(x, A, B, C):
    return A * x * x + B * x + C

dataset=np.array(viirs[['logviirs','logdmsp','DNdmsp']])
x=viirs[['logdmsp','DNdmsp']]
y=viirs['DNviirs']
from sklearn.svm import SVR

# 使用线性核函数配置的支持向量机进行回归训练，并且对测试样本进行预测。
linear_svr = SVR(kernel='linear')
linear_svr.fit(x, y)
linear_svr_y_predict = linear_svr.predict(x)

# 使用多项式核函数配置的支持向量机进行回归训练，并且对测试样本进行预测。
poly_svr = SVR(kernel='poly')
poly_svr.fit(x, y)
poly_svr_y_predict = poly_svr.predict(x)

# 使用径向基核函数配置的支持向量机进行回归训练，并且对测试样本进行预测。
rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(x, y)
rbf_svr_y_predict = rbf_svr.predict(x)
# 使用R-squared、MSE和MAE指标对三种配置的支持向量机（回归）模型在相同测试集上进行性能评估。
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
print('R-squared value of linear SVR is', linear_svr.score(poly_svr_y_predict, y))
#print 'The mean squared error of linear SVR is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict))
#print 'The mean absoluate error of linear SVR is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict))

# 三次曲线方程
def f_3(x, A, B):
    return A *B**x

from sklearn.linear_model import LinearRegression
model1=LinearRegression()
model1.fit(x,y)
ytest=model1.predict(x)

from sklearn.metrics import r2_score
print(model1.score(ytest,y))
print(np.mean((ytest-y)**2))
print(r2_score(y,ytest,multioutput='raw_values'))
print(model1.coef_)
from sklearn.metrics import r2_score
