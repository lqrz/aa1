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
discount = 0.9
threshold = 0.00001
'''
Do not edit below here
'''

start_time = time.time()
g = PredatorGame((0,0), (5,5), (11,11))

print "\n----------------\nExample: Iterative Policy Evaluation\n----------------"           

V, c = g.iterativePolicyEvaluation(discount, threshold)

targets = [((0,0), (5,5)), ((2,3), (5,4)), ((2,10), (10,0)), ((10,10), (0,0))]

for ex in targets:
    print str(ex) +  ": "  + str(V[g.getState(*ex)])
print "Iterations: " + str(c)

elapsed_time = time.time() - start_time
print("Elapsed time: " + str(elapsed_time))