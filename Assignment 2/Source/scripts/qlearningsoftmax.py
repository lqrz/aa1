import sys
import matplotlib.pyplot as plt
import numpy as np
import time
usleep = lambda x: time.sleep(x/1000000.0)

sys.path.append("..")
sys.path.append("./")

import predatorgame as pg

def getResults(samples, episodes, discount, tau, alpha, initValue, softmax, theta=0.00001):
	results = np.zeros((samples, episodes))
	allRMSs = np.zeros((samples, episodes))
	optimalV, c, optimalPolicy = game.valueIteration(discount, theta)
	for i in range(samples):
		Q, counts, policy, RMSs = game.Qlearning(discount, tau, episodes, alpha, initValue, softmax, optimalV)
		results[i] = counts
		allRMSs[i] = RMSs

	#for i in range(len(counts)):
	#	print str(counts[i]) + " -- " + str(explores[i])

	average = np.mean(results, axis=0)
	avgRMS = np.mean(allRMSs, axis=0)

	return average, avgRMS


game = pg.PredatorGame((0,0), (5,5), (11,11))

# parameters
samples = 10
episodes = 1000
discount = 0.7
#tau = 0.1
alpha = 0.1
initValue = 15
theta=0.00001
softmax = True


plt.ion()
plt.xlabel('Episodes')
plt.ylabel('Averaged number of steps')
for tau in [0.1,0.3,0.9,10,20]:
	results, avgRMS = getResults(samples, episodes, discount, tau, alpha, initValue, softmax, theta)
	plt.plot(results, label=r"$\tau = "+str(tau)+"$")
	plt.draw()
	plt.legend()
	plt.pause(0.0001)
	usleep(10)

plt.ioff()
plt.show()