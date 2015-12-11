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

def getResults(samples, episodes, discount, epsilon, decay):
    results = np.zeros((samples, episodes))
    #allRMSs = np.zeros((samples, episodes))
    
    #rewards = np.zeros((samples, episodes))
    
    for i in range(samples):
        print "sample: " + str(i)
        policy, totalSteps, V = game.minimaxQ(epsilon, discount, decay, episodes)
        results[i] = totalSteps
        #rewards[i] = randomReturnValues['rewards']
        


    #for i in range(len(counts)):
    #    print str(counts[i]) + " -- " + str(explores[i])

    average = np.mean(results, axis=0)
    #avgRMS = np.mean(allRMSs, axis=0)
    
    #statsDict = dict()
    #statsDict['winRatio'] = predwinsratio
    
    return average #avgRMS, statsDict

def parametersFor(category):
    result = []
    if category == 'epsilon':
        result = [0.2, 0.5] #0.05,0.1,0.3,0.9]
    elif category == 'decay':
        result = [0.7, 0.9]
    elif category == 'discount':
        result = [0.1,0.9]
    return result

# parameters

samples = 1
episodes = 1000
discount = 0.4
epsilon = 0.2
decay = np.power(10,np.log(0.01)/episodes)


skip = False
category = 'discount'
smoothing = False

game = pg.PredatorGame([(0, 0)], (5,5), (6,6))

if not skip:


    results = dict()
    randomReturnValues = dict()
    if category == 'epsilon':
        for epsilon in parametersFor(category):
            results[epsilon] = getResults(samples, episodes, discount, epsilon, decay)
    elif category == 'decay':
        for decay in parametersFor(category):
            results[decay] = getResults(samples, episodes, discount, epsilon, decay)
    elif category == 'discount':
        for discount in parametersFor(category):
            print(discount)
            results[discount] = getResults(samples, episodes, discount, epsilon, decay)
    else:
        sys.exit()
    print(results)
    results['episode'] = range(1,episodes+1)

    dataF = DataFrame(results)
    dataF.to_pickle('data/'+str(episodes)+category+"small")
    #pickle.dump(randomReturnValues, open('data/values'+str(episodes)+category+str(softmax), 'w+'))
else:
    dataF = pd.read_pickle('data/'+str(episodes)+category)
    #randomReturnValues = pickle.load(open('data/values'+str(episodes)+category+str(softmax), 'r+'))

print dataF
if smoothing:
    for par in parametersFor(category):
        dataF[par] = scipy.ndimage.filters.gaussian_filter(dataF[par],5*(episodes/4000),0)
episodeData = pd.melt(dataF, id_vars=['episode'], var_name=category)

ylabel = "Steps"

p = ggplot(episodeData, aes('episode', 'value', color=category)) +\
    geom_line(alpha=0.6) +\
    theme_bw() + theme() + ylab(ylabel) + xlab("Episodes") #+ ylim(0,1)
print(p)
ggsave(p, "plots/"+str(episodes)+category+"small.png")
ggsave(p, "plots/"+str(episodes)+category+"small.pdf")
