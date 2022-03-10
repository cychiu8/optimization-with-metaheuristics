import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

Dist = pd.DataFrame([[0,1,2,3,1,2,3,4],[1,0,1,2,2,1,2,3],[2,1,0,1,3,2,1,2],
                      [3,2,1,0,4,3,2,1],[1,2,3,4,0,1,2,3],[2,1,2,3,1,0,1,2],
                      [3,2,1,2,2,1,0,1],[4,3,2,1,3,2,1,0]],
                    columns=["A","B","C","D","E","F","G","H"],
                    index=["A","B","C","D","E","F","G","H"])

Flow = pd.DataFrame([[0,5,2,4,1,0,0,6],[5,0,3,0,2,2,2,0],[2,3,0,0,0,0,0,5],
                      [4,0,0,0,5,2,2,10],[1,2,0,5,0,10,0,0],[0,2,0,2,10,0,5,1],
                      [0,2,0,2,0,5,0,10],[6,0,5,10,0,1,10,0]],
                    columns=["A","B","C","D","E","F","G","H"],
                    index=["A","B","C","D","E","F","G","H"])

T0 = 1500
Temperature = T0
temp_for_plot = T0
M = 250
N = 20
alpha = 0.9

initialSolution = ["B","D","A","E","C","F","G","H"]
temp = []
obj_val=[]


def objectiveValue(array):
    df=Dist.reindex(columns=array, index=array)
    arr = np.array(df)
    start = pd.DataFrame(arr*Flow)
    objarr = np.array(start)
    obj  = sum(sum(objarr))
    return obj

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

def generateNeighbor(current):
    """
    swap?
    """
    ran_1 = np.random.randint(0,len(current))
    ran_2 = np.random.randint(0,len(current))
    while ran_1==ran_2:
            ran_2 = np.random.randint(0,len(current))
    neighbor = []
    for idx, val in enumerate(current):
        if val == current[ran_1]:
            neighbor.append(current[ran_2])
        elif val == current[ran_2]:
            neighbor.append(current[ran_1])
        else:
            neighbor.append(current[idx])
    return neighbor

def printSolution():
    print('Solution = =', current)
    print('obj = %0.3f' %obj_val_current)

current =  ["B","D","A","E","C","F","G","H"]
obj_val_current = objectiveValue(current)

for i in range(M):
    
    for j in range(N):
        
        # search the neighbor
        neighbor = generateNeighbor(current)

        obj_val_neighbor = objectiveValue(neighbor)
        obj_val_current = objectiveValue(current)
        
        if (acceptNeighbor(obj_val_neighbor, obj_val_current)):
            current = neighbor
        
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