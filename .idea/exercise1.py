import numpy as np
from matplotlib import pyplot as plt

xvalue = []
yvalue = []

print("s = set points, p = plot points")

while(1):

    key = input(': ')


    if key == 's':
        x = int(input('x: '))
        y = int(input('y: '))
        xvalue.append(x)
        yvalue.append(y)


    elif key == 'p':

        #print(xvalue)
        #print(yvalue)
        plt.plot([xvalue],[yvalue], 'ro')

        plt.axis([0,20, 0,25])
        plt.show()
