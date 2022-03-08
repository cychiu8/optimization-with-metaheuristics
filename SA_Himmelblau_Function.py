import numpy as np
import matplotlib.pyplot as plt
import random

T0 = 1000
Temperature = T0
temp_for_plot = T0
M = 300
N = 15
alpha = 0.85
k = 0.1

temp = []
obj_val = []

def generateStep():
    direction = np.random.rand()
    step = np.random.rand()
    if direction >= 0.5:
        return k*step
    return -k*step

def objectiveValue(x, y):
    return ((x**2) + y - 11)**2 +(x+(y**2) - 7)**2

def tempatureFormula(neighbor, current):
    return 1/(np.exp((neighbor - current)/Temperature))

def isObjBetter(neighbor, current):
    return obj_val_neighbor <= obj_val_current

def acceptNeighbor(neighbor, current):
    if (isObjBetter(neighbor, current)):
        return True
    rand_threshold = np.random.rand()
    if rand_threshold <= tempatureFormula(neighbor,current):
        return True
    return False

def printSolution():
    print('x = %0.3f' %x)
    print('y = %0.3f' %y)
    print('obj = %0.3f' %obj_val_current)

x= 2
y = 1
obj_val_current = ((x**2) + y - 11)**2 +(x+(y**2) - 7)**2

for i in range(M):
    
    for j in range(N):
        
        # search the neighbor
        x_neighbor = x + generateStep()
        y_neighbor = y + generateStep()
        
        obj_val_neighbor = objectiveValue(x_neighbor, y_neighbor)
        obj_val_current = objectiveValue(x, y)
        
        if (acceptNeighbor(obj_val_neighbor, obj_val_current)):
            x =  x_neighbor
            y =  y_neighbor
        
    temp.append(Temperature)
    obj_val.append(obj_val_current)
    Temperature = alpha*Temperature
            
        
printSolution()

plt.plot(temp,obj_val)
plt.title("Z at Temperature Values",fontsize=20, fontweight='bold')
plt.xlabel("Temperature",fontsize=18, fontweight='bold')
plt.ylabel("Z",fontsize=18, fontweight='bold')

plt.xlim(temp_for_plot,0)
plt.xticks(np.arange(min(temp),max(temp),100),fontweight='bold')
plt.yticks(fontweight='bold')
plt.show()