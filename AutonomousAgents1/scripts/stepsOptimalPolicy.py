import sys
import matplotlib.pyplot as plt
import numpy as np
import time
usleep = lambda x: time.sleep(x/1000000.0)

sys.path.append("..")
sys.path.append("./")

import predatorgame as pg

# parameters
episodes = 1000
discount = 0.7
epsilon = 0.1
alpha = 0.1
initValue = 15
softmax = False

tries = 10

a = pg.PredatorGame((0,0), (5,5), (11,11))

def getQlearningOptimalPolicyMeanNumberOfSteps():

	a.reset()

	for i in range(tries):
		steps = []
		Q, counts, optimalPolicy = a.Qlearning(discount, epsilon, episodes, alpha, initValue, softmax)
		a.reset()

		count = 0
		while (not a.state):
			count += 1
			#a.step(optimalPolicy[a.getState(a.predCoord, a.preyCoord)])
			a.step(optimalPolicy)
		steps.append(count)

	return np.mean(steps, axis=0)

def getSarsaOptimalPolicyMeanNumberOfSteps():

	a.reset()

	for i in range(tries):
		steps = []
		Q, counts, optimalPolicy = a.sarsa(discount, alpha, epsilon, episodes, initValue, softmax)
		a.reset()

		count = 0
		while (not a.state):
			count += 1
			#a.step(optimalPolicy[a.getState(a.predCoord, a.preyCoord)])
			a.step(optimalPolicy)
		steps.append(count)

	return np.mean(steps, axis=0)

QlearningOptimalPolicySteps = getQlearningOptimalPolicyMeanNumberOfSteps()
SarsaOptimalPolicySteps = getSarsaOptimalPolicyMeanNumberOfSteps()

print 'Qlearning optimal policy steps: ' + str(QlearningOptimalPolicySteps)
print 'Sarsa optimal policy steps: ' + str(SarsaOptimalPolicySteps)