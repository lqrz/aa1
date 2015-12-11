import sys
import numpy as np
import time
import matplotlib.pyplot as plt
sys.path.append("..")
sys.path.append("./")
from predatorgame import PredatorGame

'''
Edit parameters
'''
discount = 0.1
threshold = 0.00001
'''
Do not edit below here
'''

start_time = time.time()
g = PredatorGame((0,0), (5,5), (11,11))

print "\n----------------\nExample: Value Iteration\n----------------"        

preyPosition = (5,5)

V, c, policy = g.valueIteration(discount, threshold)

example = []

# 
preds = [(a, b) for a in range(11) for b in range(11)]
for p in preds:
    example.append((p, preyPosition))


valuesMatrix = np.zeros((g.boardSize[0], g.boardSize[1]))

for e in example:
    print e
    valuesMatrix[e[0][0]][e[0][1]] = round(V[g.getState(*e)], 4)


g.checkerboard_table(valuesMatrix, preyPosition)
plt.show()
elapsed_time = time.time() - start_time
print "Iterations: " + str(c)
print("Elapsed time: " + str(elapsed_time))