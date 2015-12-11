import sys
import numpy as np
import time
sys.path.append("..")
sys.path.append("./")
from predatorgame import PredatorGame

print "\n----------------\nExample: Random Policy\n----------------"
start_time = time.time()
g = PredatorGame((0,0), (5,5), (11,11))
results = []
for i in range(100):
    g = PredatorGame((0,0), (5,5), (11,11))
    c = 0
    while (not g.state):
             c += 1
             g.step()
             #print(str(g.predCoord) + " -- " + str(g.preyCoord))

    results.append(c)        # c = 0

mean = np.mean(results)
stdv = np.std(results)
print "Min: " + str(min(results))
print "Max: " + str(max(results))
print "Mean: " + str(mean)
print "Standard deviation: " + str(stdv)
elapsed_time = time.time() - start_time
print("Elapsed time: " + str(elapsed_time))