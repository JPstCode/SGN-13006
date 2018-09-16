import numpy as np
from matplotlib import pyplot as plt
from scipy import stats


xvalues = [2, 3, 4, 2, 4, 1, 1, 1, 2, 4]
yvalues = [1, 1, 1, 2, 2, 3, 4, 5, 4, 5]
n = len(xvalues)


a1 = np.sum(np.prod([xvalues,yvalues],axis=0)/n)
a2 = (np.sum(xvalues)/n)*(np.sum(yvalues)/n)
a3 = np.sum(np.power(xvalues,2))/n
a4 = np.power(np.sum(xvalues)/n,2)

a = (a1-a2)/(a3-a4)

b1 = np.sum(yvalues)/n
b2 = np.sum(xvalues)/n

b = b1 - b2*a

abline_val = [a * i + b for i in xvalues]

plt.figure(1)
plt.title("calc")
plt.plot(xvalues,yvalues,'ro')
plt.plot(xvalues,abline_val, 'b', label= 'calc')




slope, intercept, _,_,_ = stats.linregress(xvalues,yvalues)
abline_lin = [slope * i + intercept for i in xvalues]

print(a, " ",b)
print(slope," ",intercept)

plt.figure(2)
plt.title("scipy")
plt.plot(xvalues,yvalues,'ro')
plt.plot(xvalues, abline_lin, 'r', label= 'scipy')

plt.show()