from ggplot import *
import sys
import numpy as np
import time
import pandas as pd
from pandas import DataFrame
usleep = lambda x: time.sleep(x/1000000.0)

sys.path.append("..")
sys.path.append("./")

import predatorgame as pg

def getResults(samples, episodes, discount, epsilon, alpha, initValue, softmax):
	results = np.zeros((samples, episodes))
	optimalV, c, optimalPolicy = game.valueIteration(discount, theta)
	for i in range(samples):
		Q, counts, policy, RMSs = game.Qlearning(discount, epsilon, episodes, alpha, initValue, softmax)
		results[i] = counts

	#for i in range(len(counts)):
	#	print str(counts[i]) + " -- " + str(explores[i])

	average = np.mean(results, axis=0)

	return average

# parameters
samples = 1
episodes = 1000
discount = 0.7
epsilon = 0.1
alpha = 0.1
initValue = 15
theta=0.00001
softmax = False
skip = False

game = pg.PredatorGame((0,0), (5,5), (11,11))
if not skip:


	results = dict()
	for initValue in [0, 1, 10, 15]:
		results[initValue] = getResults(samples, episodes, discount, epsilon, alpha, initValue, softmax)
		print initValue
	results['episode'] = range(0,episodes)
	dataF = DataFrame(results)
	dataF.to_pickle('data/initValues'+str(episodes))
else:
	dataF = pd.read_pickle('data/initValues'+str(episodes))

episodeData = pd.melt(dataF, id_vars=['episode'], var_name='initValue')

# plt.ioff()
#x = qplot(range(0,4), [0.68834, 0.76024, 0.82407, 0.82113], geom = ["point", "line"])
#print x
# print qplot([0,1], [0.68834, 0.76024])
p = ggplot(episodeData, aes('episode', 'value', color='initValue')) +\
    geom_line() +\
    theme_bw() + theme() + ylab("Steps") + xlab("Episodes") + ylim(0,60)
print p
ggsave(p, "plots/initValues"+str(episodes)+".png")
ggsave(p, "plots/initValues"+str(episodes)+".pdf")
