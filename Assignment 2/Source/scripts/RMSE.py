import sys
import numpy as np
from pandas import DataFrame
import pandas as pd

sys.path.append("..")
sys.path.append("./")

import predatorgame as pg
from ggplot import *


##
def sarsaresults(samples, episodes, discount, epsilon, alpha, initValue, softmax, theta=0.00001):
    results = np.zeros((samples, episodes))
    allRMSs = np.zeros((samples, episodes))
    optimalV, c, optimalPolicy = game.valueIteration(discount, theta)
    for i in range(samples):
        Q, counts, policy, RMSs = game.sarsa(discount, alpha, epsilon, episodes, initValue, softmax, optimalV)
        results[i] = counts
        
        allRMSs[i] = RMSs

    #for i in range(len(counts)):
    #    print str(counts[i]) + " -- " + str(explores[i])

    average = np.mean(results, axis=0)
    avgRMS = np.mean(allRMSs, axis=0)
    
    return average, avgRMS


def montecarloOnPolicyresults(samples, episodes, discount, epsilon, initValue, theta=0.00001):
    allstepRatios = np.zeros((samples, episodes))
    allRMSs = np.zeros((samples, episodes))
    allstepCounts = np.zeros((samples, episodes))

    optimalV, c, optimalPolicy = game.valueIteration(discount, theta)
    
    for i in range(samples):
        Q, stepCounts, stepRatios, RMSs = game.monteCarloControl(discount, epsilon, episodes, initValue, optimalV, optimalPolicy)
        
        allstepCounts[i] = stepCounts
        allstepRatios[i] = stepRatios
        allRMSs[i] = RMSs

    #for i in range(len(counts)):
    #    print str(counts[i]) + " -- " + str(explores[i])

    avgstepCounts = np.mean(allstepCounts, axis=0)
    avgstepRatios = np.mean(allstepRatios, axis=0)
    avgRMS = np.mean(allRMSs, axis=0)
    
    return avgstepCounts, avgstepRatios, avgRMS

def montecarloOffPolicyresults(samples, episodes, discount, epsilon, initValue, theta=0.00001):
    allstepRatios = np.zeros((samples, episodes))
    allRMSs = np.zeros((samples, episodes))
    alloptimalStepRatios = np.zeros((samples, episodes))
    allstepCounts = np.zeros((samples, episodes))

    optimalV, c, optimalPolicy = game.valueIteration(discount, theta)

    for i in range(samples):
        Q, stepCounts, stepRatios, optimalStepRatios, RMSs = game.offPolicyMonteCarloControl(discount, epsilon, episodes, initValue, optimalV, optimalPolicy)
        
        allstepCounts[i] = stepCounts
        allstepRatios[i] = stepRatios
#         alloptimalStepRatios[i] = optimalStepRatios
        allRMSs[i] = RMSs

    #for i in range(len(currentSampleCounts)):
    #    print str(currentSampleCounts[i]) + " -- " + str(explores[i])
    
    avgstepCounts = np.mean(allstepCounts, axis=0)
    avgstepRatios = np.mean(allstepRatios, axis=0)
#     avgoptimalStepRatios = np.mean(alloptimalStepRatios, axis=0)
    avgoptimalStepRatios = 0
    avgRMS = np.mean(allRMSs, axis=0)

    return avgstepCounts, avgstepRatios, avgoptimalStepRatios, avgRMS

def qlearningsoftmaxresults(samples, episodes, discount, tau, alpha, initValue, softmax, theta=0.00001):
    results = np.zeros((samples, episodes))
    allRMSs = np.zeros((samples, episodes))
    optimalV, c, optimalPolicy = game.valueIteration(discount, theta)
    for i in range(samples):
        Q, counts, policy, RMSs = game.Qlearning(discount, tau, episodes, alpha, initValue, softmax, optimalV)
        results[i] = counts
        allRMSs[i] = RMSs

    #for i in range(len(counts)):
    #    print str(counts[i]) + " -- " + str(explores[i])

    average = np.mean(results, axis=0)
    avgRMS = np.mean(allRMSs, axis=0)

    return average, avgRMS

def qlearningresults(samples, episodes, discount, epsilon, alpha, initValue, softmax, theta=0.00001):
    results = np.zeros((samples, episodes))
    allRMSs = np.zeros((samples, episodes))
    optimalV, c, optimalPolicy = game.valueIteration(discount, theta)
    for i in range(samples):
        Q, counts, policy, RMSs = game.Qlearning(discount, epsilon, episodes, alpha, initValue, softmax, optimalV)
        results[i] = counts
        allRMSs[i] = RMSs

    #for i in range(len(counts)):
    #    print str(counts[i]) + " -- " + str(explores[i])

    average = np.mean(results, axis=0)
    avgRMS = np.mean(allRMSs, axis=0)

    return average, avgRMS
##

rmse = dict()

# parameters
samples = 1
episodes = 3000
discount = 0.7
epsilon = 0.1
alpha = 0.1
tau = 0.1
initValue = 15
theta = 0.00001
softmax = False
skip = True

if not skip:
    game = pg.PredatorGame((0,0), (5,5), (11,11))

    notused, rmse['Sarsa'] = sarsaresults(samples, episodes, discount, epsilon, alpha, initValue, softmax, theta)
    notused, rmse['Q-learning'] = qlearningresults(samples, episodes, discount, epsilon, alpha, initValue, softmax, theta)
    notused, rmse['Q-learning with SoftMax'] = qlearningsoftmaxresults(samples, episodes, discount, tau, alpha, initValue, softmax, theta)
    notused, notused, rmse['On Policy Monte Carlo'] = montecarloOnPolicyresults(samples, episodes, discount, epsilon, 0, theta)
    notused, notused, notused, rmse['Off Policy Monte Carlo'] = montecarloOffPolicyresults(samples, episodes, discount, epsilon, 0, theta)

    rmse['episode'] = range(0,episodes)
    dataF = DataFrame(rmse)
    dataF.to_pickle('data/rmse'+str(episodes))
else:
    dataF = pd.read_pickle('data/rmse'+str(episodes))

episodeData = pd.melt(dataF, id_vars=['episode'], var_name='Learning algorithm')
# for key, value in rmse.items():
#     plt.figure()
#     plt.plot(value, 'b')
#     plt.xlabel('Episodes')
#     plt.ylabel('Root Mean Square Error ('+key+')')
#     plt.legend()

# plt.show()   

p = ggplot(episodeData, aes('episode', 'value', color='Learning algorithm')) +\
    geom_line() +\
    theme_bw() + theme() + ylab("RMSE") + xlab("Episodes") + ylim(0,2)
print p
ggsave(p, "plots/rmse"+str(episodes)+".png")
ggsave(p, "plots/rmse"+str(episodes)+".pdf")
