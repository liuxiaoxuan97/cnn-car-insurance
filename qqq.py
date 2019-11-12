import numpy as np

import pandas as pd
viirs=pd.read_csv("C:/Users/lenovo/Desktop/viirsdmps1.csv",encoding='gb2312')
print(viirs.head(3))
import matplotlib.pyplot as plt

viirs['logviirs']=np.log(viirs['DNviirs'])
viirs['logdmsp']=np.log(viirs['DNdmsp'])
data=viirs[['DNviirs','logdmsp']]


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def func(x, a, b,c):
    return a * np.log(b*x)+c


# Define the data to be fit with some noise:
xdata = np.array(viirs.DNdmsp)
print(xdata)

y = np.array(viirs.DNviirs)

ydata = y
plt.scatter(xdata, ydata, label='data')

# Fit for the parameters a, b, c of the function func:
popt, pcov = curve_fit(func, xdata, ydata)
print(popt)
plt.plot(xdata, func(xdata, *popt), 'r-')
         #label='fit: a=%f, b=%f, c=%f' % tuple(popt))

# Constrain the optimization to the region of 0 <= a <= 3, 0 <= b <= 1 and 0 <= c <= 0.5:
popt, pcov = curve_fit(func, xdata, ydata)
print(popt[0])
plt.plot(xdata, func(xdata, *popt), 'g--')
         #label='fit: a=%f, b=%f, c=%f' % tuple(popt))

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
xx=viirs.DNdmsp
yhat=popt[0]* np.log(popt[1]*xx)+popt[2]
ymean=ydata.mean()
#print(ydata-ymean)
print(sum((yhat-ydata)**2)/sum((ydata-ymean)**2))
print(sum((yhat-ydata)**2)/358)
data1=pd.read_csv("C:/Users/lenovo/Desktop/dat.csv",encoding='gb2312')
#print(data1.head(5))
yhat=popt[0]* np.log(popt[1]*xx)+popt[2]

data1['1997re']=popt[0]* np.log(popt[1]*data1['1997'])+popt[2]
data1['1998re']=popt[0]* np.log(popt[1]*data1['1998'])+popt[2]
data1['1999re']=popt[0]* np.log(popt[1]*data1['1999'])+popt[2]
data1['2000re']=popt[0]* np.log(popt[1]*data1['2000'])+popt[2]
data1['2001re']=popt[0]* np.log(popt[1]*data1['2001'])+popt[2]
data1['2002re']=popt[0]* np.log(popt[1]*data1['2002'])+popt[2]
data1['2003re']=popt[0]* np.log(popt[1]*data1['2003'])+popt[2]
data1['2004re']=popt[0]* np.log(popt[1]*data1['2004'])+popt[2]
data1['2005re']=popt[0]* np.log(popt[1]*data1['2005'])+popt[2]
data1['2006re']=popt[0]* np.log(popt[1]*data1['2006'])+popt[2]
data1['2007re']=popt[0]* np.log(popt[1]*data1['2007'])+popt[2]
data1['2008re']=popt[0]* np.log(popt[1]*data1['2008'])+popt[2]
data1['2009re']=popt[0]* np.log(popt[1]*data1['2009'])+popt[2]
data1['2010re']=popt[0]* np.log(popt[1]*data1['2010'])+popt[2]
data1['2011re']=popt[0]* np.log(popt[1]*data1['2011'])+popt[2]
data1['2012re']=popt[0]* np.log(popt[1]*data1['2012'])+popt[2]
print(data1.head(2))
data1.to_csv("C:/Users/lenovo/Desktop/loghou.csv",sep=',')
m=1.81445
p=27.38141
q=8.45108

#data1.to_csv("C:/Users/lenovo/Desktop/data3erci.csv",sep=',')
datadmsp=pd.read_csv("C:/Users/lenovo/Desktop/data1.csv",sep=',',encoding='gb2312')
datafinal=pd.merge(data1,datadmsp,on='Prefecture',how='outer')
datamingzi=pd.read_excel("C:/Users/lenovo/Desktop/市级平均数据.xlsx",sheet='Average',encoding='gb2312')
print(datamingzi.head(3))
mingzi=datamingzi[['Prefecture','English']]
datafinall=pd.merge(datafinal,mingzi,on='Prefecture',how='outer')
datafinall.to_csv("C:/Users/lenovo/Desktop/datafinall.csv",sep=',')