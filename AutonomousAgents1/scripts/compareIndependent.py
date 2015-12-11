import sys
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.ndimage.filters

import pandas as pd
from pandas import DataFrame
from ggplot import *

sys.path.append("..")
sys.path.append("./")

import predatorgame as pg

# parameters
episodes = 50000
discount = 0.7
epsilon = 0.1
alpha = 0.1
initValue = 0
softmax = False
samples = 1
skip = True
game = pg.PredatorGame([(0,0), (10,10)], (5,5), (11,11))

def getIndependentQLearning():
	results = np.zeros((samples, episodes))
	rewards = np.zeros((samples, episodes))

	for i in range(samples):
		print i
		Q, Qprey, counts, policy, policyPrey, randomReturnValues = game.IndependentQlearning(discount, epsilon, episodes, alpha, initValue, softmax)
		results[i] = counts
		rewards[i] = randomReturnValues['rewards']


	predwinsratio = np.sum(rewards * (rewards > 0) / 10, axis=0) * 1.0 / samples

	average = np.mean(results, axis=0)

	return average, predwinsratio

def getIndependentSarsa():
	results = np.zeros((samples, episodes))
	rewards = np.zeros((samples, episodes))

	for i in range(samples):
		print i
		Q, Qprey, counts, policy, policyPrey, randomReturnValues = game.independentSarsa(discount, alpha, epsilon, episodes, initValue, softmax)
		results[i] = counts
		rewards[i] = randomReturnValues['rewards']

	predwinsratio = np.sum(rewards * (rewards > 0) / 10, axis=0) * 1.0 / samples

	average = np.mean(results, axis=0)

	return average, predwinsratio

data = dict()
alg = ['IndependentQLearning', 'IndependentSarsa']

N = 1
if not skip:
	for i in range(N):
		print i

		averageQ, predwinsratioQ = getIndependentQLearning()

		averageS, predwinsratioS = getIndependentSarsa()

	data['IndependentQLearning'] = predwinsratioQ
	data['IndependentSarsa'] = predwinsratioS
	data['episode'] = range(1,episodes+1)

	dataF = DataFrame(data)
	dataF.to_pickle('data/comparison')
else:
    dataF = pd.read_pickle('data/comparison')
       
for a in alg:
	dataF[a] = scipy.ndimage.filters.gaussian_filter(dataF[a],5*(episodes/4000),0)
    

episodeData = pd.melt(dataF, id_vars=['episode'], var_name='Algorithm')

p = ggplot(episodeData, aes('episode', 'value', color='Algorithm')) +\
	 geom_line() +\
     theme_bw() + theme() + ylab("Win ratio") + xlab("Episodes")
print p
ggsave(p, "plots/comparison.png")
ggsave(p, "plots/comparison.pdf")