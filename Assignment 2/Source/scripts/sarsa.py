import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("..")
sys.path.append("./")

import predatorgame as pg

def getResults(samples, episodes, discount, epsilon, alpha, initValue, softmax, theta=0.00001):
	results = np.zeros((samples, episodes))
	allRMSs = np.zeros((samples, episodes))
	optimalV, c, optimalPolicy = game.valueIteration(discount, theta)
	for i in range(samples):
		Q, counts, policy, RMSs = game.sarsa(discount, alpha, epsilon, episodes, initValue, softmax, optimalV)
		results[i] = counts
		
		allRMSs[i] = RMSs

	#for i in range(len(counts)):
	#	print str(counts[i]) + " -- " + str(explores[i])

	average = np.mean(results, axis=0)
	avgRMS = np.mean(allRMSs, axis=0)
	
	return average, avgRMS


game = pg.PredatorGame((0,0), (5,5), (11,11))

# parameters
samples = 1
episodes = 100
discount = 0.7
epsilon = 0.1
alpha = 0.1
initValue = 15
theta = 0.00001
softmax = False


results1, avgRMS = getResults(samples, episodes, discount, 0.1, alpha, initValue, softmax, theta)
results2, avgRMS = getResults(samples, episodes, discount, 0.9, alpha, initValue, softmax, theta)



plt.plot(results1, label=r"$\epsilon = 0.1$")
plt.plot(results2, label=r"$\epsilon = 0.7$")
plt.xlabel('Episodes')
plt.ylabel('Number of steps')
plt.legend()

plt.figure()

plt.plot(avgRMS, 'b')
plt.xlabel('Episodes')
plt.ylabel('Root Mean Square Error')
plt.legend()

plt.show()
