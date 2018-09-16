from matplotlib import pyplot as plt
import numpy as np

draw = np.arange(0,10,0.5)

xvalues = []
yvalues = []

plt.axis([0,10, 0,10])

print("click 5 times")
x = plt.ginput(5)
n = len(x)

for points in x:
    i = 0
    for point in points:
        if i % 2 == 0:
            xvalues.append(point)
        else:
            yvalues.append(point)
        i = i+1


a1 = np.sum(np.prod([xvalues,yvalues],axis=0)/n)
a2 = (np.sum(xvalues)/n)*(np.sum(yvalues)/n)
a3 = np.sum(np.power(xvalues,2))/n
a4 = np.power(np.sum(xvalues)/n,2)

a = (a1-a2)/(a3-a4)

b1 = np.sum(yvalues)/n
b2 = np.sum(xvalues)/n

b = b1-b2*a

regline = [a * i + b for i in draw]

plt.figure(2)
plt.plot(xvalues,yvalues,'ro')
plt.plot(draw,regline, 'b')
plt.show()
