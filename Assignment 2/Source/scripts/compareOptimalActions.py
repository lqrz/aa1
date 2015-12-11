import sys
import matplotlib.pyplot as plt
import numpy as np
import time

import pandas as pd
from pandas import DataFrame
#from ggplot import *

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
theta = 0.00001
softmax = False

tries = 10

game = pg.PredatorGame((0,0), (5,5), (11,11))

optimalV, c, optimalPolicy = game.valueIteration(discount, theta)

def getValueIterationPolicy():
	#Value iteration Params
	discount = 0.1
	threshold = 0.00001

	game.reset()

	V, c, policy = game.valueIteration(discount, threshold)

	return policy

def getQLearningPolicy():
	#QLearning params
	discount = 0.7
	epsilon = 0.1
	episodes = 1000
	alpha = 0.1
	initValue = 0
	softmax = False

	game.reset()

	Q, counts, policy, rsme = game.Qlearning(discount, epsilon, episodes, alpha, initValue, softmax, optimalV)
	
	return policy

def getSarsaPolicy():
	#Sarsa params
	episodes = 1000
	discount = 0.7
	epsilon = 0.1
	alpha = 0.1
	initValue = 0
	softmax = False

	game.reset()

	Q, counts, policy, rsme = game.sarsa(discount, alpha, epsilon, episodes, initValue, softmax, optimalV)

	return policy

def getOffPolicyMonteCarlo():
	pass

def getOnPolicyMonteCarlo():
	pass

def compareAlgorithms(comparedPolicy, controlPolicy=None):
	# compares comparedPolicy to controlPolicy.
	# returns number of steps till end + number of optimal actions
	
	compareCount = 0
	stepsCount = 0

	game.reset()

	while(not game.state):
		stepsCount += 1
		state = game.getState(game.predCoord, game.preyCoord)
		if controlPolicy and comparedPolicy[state] == controlPolicy[state]:
			compareCount += 1
		game.step(comparedPolicy)

	return stepsCount, compareCount


sarsaPolicy = getSarsaPolicy()
offMonteCarloPolicy = getOffPolicyMonteCarlo()
onMonteCarloPolicy = getOnPolicyMonteCarlo()


qLearningStepsCounts = []
qLearningCompareCounts = []
sarsaStepsCounts = []
sarsaCompareCounts = []
valueIterationStepsCounts = []

data = dict()
# data['Off policy MC'] = [0]
# data['On policy MC'] = [0]

N = 50

print 'start looping'
for i in range(N):
	print i
	valueIterationPolicy = getValueIterationPolicy()
	valueIterationStepsCount, tmp = compareAlgorithms(valueIterationPolicy) # if there's no algorithm to compare, it will just run the given policy
	valueIterationStepsCounts.append(valueIterationStepsCount)

	qLearningPolicy = getQLearningPolicy()
	qLearningStepsCount, qLearningCompareCount = compareAlgorithms(qLearningPolicy, valueIterationPolicy)
	qLearningStepsCounts.append(qLearningStepsCount)
	qLearningCompareCounts.append(qLearningCompareCount)

	sarsaPolicy = getSarsaPolicy()
	sarsaStepsCount, sarsaCompareCount = compareAlgorithms(sarsaPolicy, valueIterationPolicy)
	sarsaStepsCounts.append(sarsaStepsCount)
	sarsaCompareCounts.append(sarsaCompareCount)

data['Value iteration'] = valueIterationStepsCounts
data['QLearning'] = qLearningStepsCounts
data['Sarsa'] = sarsaStepsCounts
data['episode'] = range(0,N)

dataF = DataFrame(data)

episodeData = pd.melt(dataF, id_vars=['episode'], var_name='algorithm')

p = ggplot(episodeData, aes('episode', 'value', color='algorithm')) +\
	 geom_line() +\
     theme_bw() + theme() + ylab("") + xlab("")
print p

print 'Finished looping'

print 'On average, value iteration took: ' + str(np.mean(valueIterationStepsCount)) + ' to end'
print 'On average, QLearning took ' + str(np.mean(qLearningStepsCounts)) + ' iterations to end the episode'
print 'The predator found ' + str(np.mean(qLearningCompareCounts)) + ' ('+ str(np.mean(qLearningCompareCounts)*100/np.mean(qLearningStepsCounts)) +'%)' + ' optimal moves'

print 'On average, Sarsa took ' + str(np.mean(sarsaStepsCounts)) + ' iterations to end the episode'
print 'The predator found ' + str(np.mean(sarsaCompareCounts)) + ' ('+ str(np.mean(sarsaCompareCounts)*100/np.mean(sarsaStepsCounts)) +'%)' + ' optimal moves while using Sarsa'

# plt.ion()
# plt.xlabel('Episodes')
# plt.ylabel('Number of steps till end')
# plt.plot(valueIterationStepsCounts, label=r"Value iteration")
# plt.plot(qLearningStepsCounts, label=r"QLearning")
# plt.plot(sarsaStepsCounts, label=r"Sarsa")
# plt.draw()
# plt.legend()
# plt.ioff()
# plt.show()