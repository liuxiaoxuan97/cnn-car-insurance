import pandas as pd
import numpy as np
data1=pd.read_excel("C:/Users/lenovo/Desktop/VIIRS中国各地级市灯光数据.xlsx",sheet='Sheet1')
data1=data1.iloc[1:,1:]
data1['DNvalue']=data1['DNvalue'].astype(float)
dmsp=pd.read_excel("C:/Users/lenovo/Desktop/DMSP中国各地级市灯光数据（校正后）.xlsx",sheet='DMSP中国各地级市灯光数据（校正后）')
data11=data1[data1['Year']=="2013"]
data11=data11[data11['DNvalue']!=0]

print((data11.DNvalue.sum()))
data2=data11.groupby('Prefecture').mean()
#data3就是汇总好的
data3=pd.pivot_table(data11,index=[u'Prefecture'],values=[u'DNvalue'],aggfunc=[np.size,np.mean,np.std]).reset_index()
print(data3.shape)
data3.columns=['Prefecture','Count','DNviirs','std']


print(data3[data3.Prefecture=='乌鲁木齐市'])
dmsp2013=dmsp[dmsp['Year']=='2013']
ntl2013=pd.merge(data3,dmsp2013,left_on='Prefecture',right_on='Prefecture')
#ntl2013.dropna(axis=0, how='any', inplace=True)
print(ntl2013['DNvalue'].isnull().value_counts())

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def func(x, a, b, c):
    return a * x**b + c


# Define the data to be fit with some noise:
xdata = np.linspace(0, 4, 50)
y = func(xdata, 2.5, 1.3, 0.5)
np.random.seed(1729)
y_noise = 0.2 * np.random.normal(size=xdata.size)
ydata = y + y_noise
plt.plot(xdata, ydata, 'b-', label='data')

# Fit for the parameters a, b, c of the function func:
popt, pcov = curve_fit(func, xdata, ydata)
print
popt
plt.plot(xdata, func(xdata, *popt), 'r-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

# Constrain the optimization to the region of 0 <= a <= 3, 0 <= b <= 1 and 0 <= c <= 0.5:
popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))
print(popt)
plt.plot(xdata, func(xdata, *popt), 'g--',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()