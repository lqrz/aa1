import sys
import matplotlib.pyplot as plt
import numpy as np
import time

import pandas as pd
from pandas import DataFrame
from ggplot import *

sys.path.append("..")
sys.path.append("./")

import predatorgame as pg

# parameters
episodes = 100
discount = 0.7
epsilon = 0.1
alpha = 0.1
initValue = 15
softmax = False

game = pg.PredatorGame((0,0), (5,5), (11,11))


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
	alpha = 0.1
	initValue = 15
	softmax = False

	game.reset()
	
	Q, counts, policy = game.Qlearning(discount, epsilon, episodes, alpha, initValue, softmax)
	
	return policy, counts

def getSarsaPolicy():
	#Sarsa params
	discount = 0.7
	epsilon = 0.1
	alpha = 0.1
	initValue = 15
	softmax = False

	game.reset()

	Q, counts, policy = game.sarsa(discount, alpha, epsilon, episodes, initValue, softmax)

	return policy, counts

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

N = 20

qLearningStepsToLearn = np.zeros((N, episodes))
sarsaStepsToLearn = np.zeros((N, episodes))

print 'start looping'
for i in range(N):
	print i
	valueIterationPolicy = getValueIterationPolicy()
	valueIterationStepsCount, tmp = compareAlgorithms(valueIterationPolicy) # if there's no algorithm to compare, it will just run the given policy
	valueIterationStepsCounts.append(valueIterationStepsCount)

	qLearningPolicy, steps = getQLearningPolicy()
	qLearningStepsCount, qLearningCompareCount = compareAlgorithms(qLearningPolicy, valueIterationPolicy)
	qLearningStepsCounts.append(qLearningStepsCount)
	qLearningCompareCounts.append(qLearningCompareCount)
	qLearningStepsToLearn[i] = steps

	sarsaPolicy, steps= getSarsaPolicy()
	sarsaStepsCount, sarsaCompareCount = compareAlgorithms(sarsaPolicy, valueIterationPolicy)
	sarsaStepsCounts.append(sarsaStepsCount)
	sarsaCompareCounts.append(sarsaCompareCount)
	sarsaStepsToLearn[i] = steps

qLearningStepsToLearn_mean= np.mean(qLearningStepsToLearn, axis=0)
sarsaStepsToLearn_mean = np.mean(sarsaStepsToLearn, axis=0)

data2 = dict()

data2['QLearning'] = qLearningStepsToLearn_mean
data2['Sarsa'] = sarsaStepsToLearn_mean
data2['episode'] = range(0,episodes)

dataF2 = DataFrame(data2)

episodeData2 = pd.melt(dataF2, id_vars=['episode'], var_name="algorithm")
p2 = ggplot(episodeData2, aes('episode', 'value', color='algorithm')) +\
	 geom_line() +\
     theme_bw() + theme() + ylab("") + xlab("")
print p2


#---- New graph ----
# data['Value iteration'] = valueIterationStepsCounts
# data['QLearning'] = qLearningStepsCounts
# data['Sarsa'] = sarsaStepsCounts
# data['episode'] = range(0,N)

# dataF = DataFrame(data)

# episodeData = pd.melt(dataF, id_vars=['episode'], var_name='algorithm')

# p = ggplot(episodeData, aes('episode', 'value', color='algorithm')) +\
# 	 geom_line() +\
#      theme_bw() + theme() + ylab("") + xlab("")
# print p

print 'Finished looping'

print 'On average, value iteration took: ' + str(np.mean(valueIterationStepsCounts)) + ' to end'
print 'On average, QLearning took ' + str(np.mean(qLearningStepsCounts)) + ' iterations to end the episode'
print 'The predator found ' + str(np.mean(qLearningCompareCounts)) + ' ('+ str(np.mean(qLearningCompareCounts)*100/np.mean(qLearningStepsCounts)) +'%)' + ' optimal moves'

print 'On average, Sarsa took ' + str(np.mean(sarsaStepsCounts)) + ' iterations to end the episode'
print 'The predator found ' + str(np.mean(sarsaCompareCounts)) + ' ('+ str(np.mean(sarsaCompareCounts)*100/np.mean(sarsaStepsCounts)) +'%)' + ' optimal moves while using Sarsa'

#---- Old graph----
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