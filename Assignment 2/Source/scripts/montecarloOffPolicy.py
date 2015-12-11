import sys
import matplotlib.pyplot as plt
import numpy as np
import math

sys.path.append("..")
sys.path.append("./")

import predatorgame as pg

def getResults(samples, episodes, discount, epsilon, initValue, theta=0.00001):
	allstepRatios = np.zeros((samples, episodes))
	allRMSs = np.zeros((samples, episodes))
	alloptimalStepRatios = np.zeros((samples, episodes))
	allstepCounts = np.zeros((samples, episodes))

	optimalV, c, optimalPolicy = game.valueIteration(discount, theta)

	for i in range(samples):
		Q, stepCounts, stepRatios, optimalStepRatios, RMSs = game.offPolicyMonteCarloControl(discount, epsilon, episodes, initValue, optimalV, optimalPolicy)
		
		allstepCounts[i] = stepCounts
		allstepRatios[i] = stepRatios
# 		alloptimalStepRatios[i] = optimalStepRatios
		allRMSs[i] = RMSs

	#for i in range(len(currentSampleCounts)):
	#	print str(currentSampleCounts[i]) + " -- " + str(explores[i])
	
	avgstepCounts = np.mean(allstepCounts, axis=0)
	avgstepRatios = np.mean(allstepRatios, axis=0)
# 	avgoptimalStepRatios = np.mean(alloptimalStepRatios, axis=0)
	avgoptimalStepRatios = 0
	avgRMS = np.mean(allRMSs, axis=0)

	return avgstepCounts, avgstepRatios, avgoptimalStepRatios, avgRMS


game = pg.PredatorGame((0,0), (5,5), (11,11))

# parameters
samples = 1
episodes = 300
discount = 0.7
epsilon = 0.1
initValue = 15
theta = 0.00001 #for value iteration

stepCounts, stepRatios, optimalStepRatios, RMSs = getResults(samples, episodes, discount, epsilon, 0)

plt.plot(RMSs, 'b')
plt.xlabel('Episodes')
plt.ylabel('Root Mean Square Error')
plt.legend()

plt.figure()

plt.plot(stepCounts, 'g')
plt.xlabel('Episodes')
plt.ylabel('Averaged number of steps')
plt.legend()
 
plt.figure()

plt.plot(stepRatios, 'k')
plt.xlabel('Episodes')
plt.ylabel('Ratio of number of steps taken to number of steps by following optimal policy')
plt.legend()
 
#plt.figure()

# plt.plot(optimalStepRatios, 'r')
# plt.xlabel('Episodes')
# plt.ylabel('Percentage optimal actions taken')
# plt.legend()

plt.show()

