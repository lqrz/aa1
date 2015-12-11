import numpy as np
import pandas as pd
from pandas import DataFrame
import sys
import pickle
import scipy.ndimage.filters

sys.path.append("..")
sys.path.append("./")

from ggplot import *
import predatorgame as pg

def getResults(samples, episodes, discount, epsilon, alpha, initValue, softmax, theta=0.00001):
    results = np.zeros((samples, episodes))
    allRMSs = np.zeros((samples, episodes))

    rewards = np.zeros((samples, episodes))

    for i in range(samples):
        print i
        Q, Qprey, counts, policy, policyPrey, randomReturnValues = game.IndependentQlearning(discount, epsilon, episodes, alpha, initValue, softmax)
        results[i] = counts
        rewards[i] = randomReturnValues['rewards']


    predwinsratio = np.sum(rewards * (rewards > 0) / 10, axis=0) * 1.0 / samples

    #for i in range(len(counts)):
    #    print str(counts[i]) + " -- " + str(explores[i])

    average = np.mean(results, axis=0)
    avgRMS = np.mean(allRMSs, axis=0)

    statsDict = dict()
    statsDict['winratio'] = predwinsratio

    return average, avgRMS, statsDict

def parametersFor(category):
    result = []
    if category == 'epsilon':
        result = [0.2] #0.05,0.1,0.3,0.9]
    elif category == 'tau':
        result = [0.05,12,30,100]
    elif category == 'alpha':
        result = [0.1,0.2,0.3,0.6,1]
    elif category == 'discount':
        result = [0.1,0.4,0.7,0.8,0.9]
    elif category == 'number of Predators':
        result = [1,2,3]
    return result

# parameters

samples = 1
episodes = 100000
discount = 0.7

epsilon = 0.1
alpha = 0.2
initValue = 15
softmax = False
skip = False
category = 'number of Predators'
smoothing = True

#graphtype = 'steps'
graphtype = 'winratio'

game = pg.PredatorGame([(0, 0), (0,10), (10, 10)], (5,5), (11,11))

if not skip:

    results = dict()
    winRatioDict = dict()
    randomReturnValues = dict()
    if category == 'epsilon':
        for epsilon in parametersFor(category):
            results[epsilon], avgRMS, randomReturnValues[epsilon] = getResults(samples, episodes, discount, epsilon, alpha, initValue, False)
            winRatioDict[epsilon] = randomReturnValues[epsilon]['winratio']
    elif category == 'tau':
        for tau in parametersFor(category):
            results[tau], avgRMS, randomReturnValues[tau] = getResults(samples, episodes, discount, epsilon, alpha, initValue, True)
            winRatioDict[tau] = randomReturnValues[tau]['winratio']
    elif category == 'alpha':
        for alpha in parametersFor(category):
            results[alpha], avgRMS, randomReturnValues[alpha] = getResults(samples, episodes, discount, epsilon, alpha, initValue, softmax)
            winRatioDict[alpha] = randomReturnValues[alpha]['winratio']
    elif category == 'discount':
        for discount in parametersFor(category):
            results[discount], avgRMS, randomReturnValues[discount] = getResults(samples, episodes, discount, epsilon, alpha, initValue, softmax)
            winRatioDict[discount] = randomReturnValues[discount]['winratio']
    elif category == 'number of Predators':
        for nPreds in parametersFor(category):
            if nPreds == 1:
                game.predCoords = game.initPredCoords = [(0,0)]
            elif nPreds == 2:
                game.predCoords = game.initPredCoords = [(0, 0), (10, 10)]
            elif nPreds == 3:
                game.predCoords = game.initPredCoords = [(0, 0), (10, 10), (0,10)]
            elif nPreds == 4:
                game.predCoords = game.initPredCoords = [(0, 0), (10, 10), (0,10), (10,0)]
            results[nPreds], avgRMS, randomReturnValues[nPreds] = getResults(samples, episodes, discount, epsilon, alpha, initValue, softmax)
            winRatioDict[nPreds] = randomReturnValues[nPreds]['winratio']
    else:
        sys.exit()

    results['episode'] = range(1,episodes+1)
    winRatioDict['episode'] = range(1,episodes+1)

    dataF_steps = DataFrame(results)
    dataF_steps.to_pickle('data/Q_steps'+str(samples)+str(episodes)+category+str(softmax))

    dataF_winratio = DataFrame(winRatioDict)
    dataF_winratio.to_pickle('data/Q_winratio'+str(samples)+str(episodes)+category+str(softmax))
else:
    dataF_steps = pd.read_pickle('data/Q_steps'+str(samples)+str(episodes)+category+str(softmax))
    dataF_winratio = pd.read_pickle('data/Q_winratio'+str(samples)+str(episodes)+category+str(softmax))

if graphtype == 'steps':
    dataToPlot = dataF_steps
    ylabel = 'Steps'
elif graphtype == 'winratio':
    dataToPlot = dataF_winratio
    ylabel = 'Win Ratio'

if smoothing:
    for par in parametersFor(category):
        dataToPlot[par] = scipy.ndimage.filters.gaussian_filter(dataToPlot[par],5*(episodes/4000),0)
episodeData = pd.melt(dataToPlot, id_vars=['episode'], var_name=category)

p = ggplot(episodeData, aes('episode', 'value', color=category)) +\
    geom_line(alpha=0.6) +\
    theme_bw() + theme() + ylab(ylabel) + xlab("Episodes")

if graphtype == 'winratio':
    p += ylim(0,1)
print(p)
ggsave(p, "plots/Q_"+graphtype+str(samples)+str(episodes)+category+str(softmax)+".png")
ggsave(p, "plots/Q_"+graphtype+str(samples)+str(episodes)+category+str(softmax)+".pdf")
