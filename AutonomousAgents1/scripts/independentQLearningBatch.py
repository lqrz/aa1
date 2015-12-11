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
        print(i)
        Q, Qprey, counts, policy, policyPrey, randomReturnValues = game.IndependentQlearning(discount, epsilon, episodes, alpha, initValue, softmax)
        results[i] = counts
        rewards[i] = randomReturnValues['rewards']
        

    predwinsratio = np.sum(rewards * (rewards > 0) / 10, axis=0) * 1.0 / samples

    #for i in range(len(counts)):
    #    print str(counts[i]) + " -- " + str(explores[i])

    average = np.mean(results, axis=0)
    avgRMS = np.mean(allRMSs, axis=0)
    
    statsDict = dict()
    statsDict['winRatio'] = predwinsratio
    
    return average, avgRMS, statsDict

def parametersFor(category):
    result = []
    if category == 'epsilon':
        result = [0.05,0.1,0.3,0.9]
    elif category == 'tau':
        result = [0.05,12,30,100]
    elif category == 'alpha':
        result = [0.1,0.2,0.3,0.6,1]
    elif category == 'discount':
        result = [0.1,0.4,0.7,0.8,0.9]
    elif category == 'winratio':
        result = ['winRatio']
    return result


predators = [[(0, 0)]] #,[(0, 0), (10, 10)],[(0, 0), (10, 10), (0,10)],
# predators = [[(0, 0), (10, 10), (0,10)],[(0, 0), (10, 10), (0,10), (10,0)]]

for predCoords in predators:
    game = pg.PredatorGame(predCoords, (5,5), (11,11))
    for category in ['epsilon','tau','alpha','discount']:
        # parameters

        samples = 1
        episodes = 100000
        discount = 0.7

        epsilon = 0.1
        alpha = 0.2
        initValue = 15
        softmax = False
        skip = False
        smoothing = True
        showGraph = True
        smoothingLevel = 15
        graphtype = 'steps'

        if not skip:

            print str(predators) + "Episodes: " + str(episodes) + "Discount: " + str(discount) + "Epsilon: " + str(epsilon) + "Alpha: " + str(alpha)
            results = dict()
            randomReturnValues = dict()
            if category == 'epsilon':
                for epsilon in parametersFor(category):
                    results[epsilon], avgRMS, randomReturnValues[epsilon] = getResults(samples, episodes, discount, epsilon, alpha, initValue, False)
            elif category == 'tau':
                for tau in parametersFor(category):
                    results[tau], avgRMS, randomReturnValues[tau] = getResults(samples, episodes, discount, epsilon, alpha, initValue, True)
            elif category == 'alpha':
                for alpha in parametersFor(category):
                    results[alpha], avgRMS, randomReturnValues[alpha] = getResults(samples, episodes, discount, epsilon, alpha, initValue, softmax)
            elif category == 'discount':
                for discount in parametersFor(category):
                    print(discount)
                    results[discount], avgRMS, randomReturnValues[discount] = getResults(samples, episodes, discount, epsilon, alpha, initValue, softmax)
            elif category == 'winratio':
                _, _, randomReturnValues['winratio'] = getResults(samples, episodes, discount, epsilon, alpha, initValue, softmax)
                results = randomReturnValues['winratio']
            else:
                sys.exit()
            print(results)
            results['episode'] = range(1,episodes+1)
            print predCoords
            dataF = DataFrame(results)
            dataF.to_pickle('data/'+str(predCoords)+str(episodes)+category+str(softmax))
            pickle.dump(randomReturnValues, open('data/values'+str(predCoords)+str(episodes)+category+str(softmax), 'w+'))
        else:
            dataF = pd.read_pickle('data/'+str(predCoords)+str(episodes)+category+str(softmax))
            randomReturnValues = pickle.load(open('data/values'+str(predCoords)+str(episodes)+category+str(softmax), 'r+'))


        ylabel = "Steps"

        ylimG = [30,180/len(predCoords)]
        if graphtype == 'steps':
            dataF = dataF
            ylabel = 'Steps'
        elif graphtype == 'winratio':
            results = dict()
            for par in parametersFor(category):
                results[par] = randomReturnValues[par]['winRatio']
            results['episode'] = range(1,episodes+1)
            dataF = DataFrame(results)
            ylabel = 'Win Ratio'
            ylimG = [0,1]

        if smoothing:
            for par in parametersFor(category):
                dataF[par] = scipy.ndimage.filters.gaussian_filter(dataF[par],smoothingLevel*(episodes/4000),0)
        episodeData = pd.melt(dataF, id_vars=['episode'], var_name=category)

        p = ggplot(episodeData, aes('episode', 'value', color=category)) +\
            geom_line(alpha=0.6) +\
            theme_bw() + theme() + ylab(ylabel) + xlab("Episodes") + ylim(ylimG[0],ylimG[1])
        if showGraph:
            print(p)
        if graphtype == 'steps':
            ggsave(p, "plots/"+str(predCoords)+str(episodes)+category+str(softmax)+".png")
            ggsave(p, "plots/"+str(predCoords)+str(episodes)+category+str(softmax)+".pdf")
        elif graphtype == 'winratio':
            ggsave(p, "plots/winratio"+str(predCoords)+str(episodes)+category+str(softmax)+".png")
            ggsave(p, "plots/winratio"+str(predCoords)+str(episodes)+category+str(softmax)+".pdf")
