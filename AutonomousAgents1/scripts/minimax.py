import numpy as np
#import pandas as pd
#from pandas import DataFrame
import sys
import numpy as np

sys.path.append("..")
sys.path.append("./")

#import ggplot
import predatorgame as pg


game = pg.PredatorGame([(0,0)], (5,5), (11,11))

nEpisodes = 200
policy, totalSteps, V = game.minimaxQ(0.2, 0.1, np.power(10,np.log(0.01)/nEpisodes), nEpisodes)

print totalSteps

for s in policy.keys():
    print str(s) + " --> " + str(policy[s])


# def getResults(samples, episodes, discount, epsilon, alpha, initValue, softmax, theta=0.00001):
#     results = np.zeros((samples, episodes))
#     allRMSs = np.zeros((samples, episodes))
#     for i in range(samples):
#         print(i)
#         Q, Qprey, counts, policy, policyPrey = game.IndependentQlearning(discount, epsilon, episodes, alpha, initValue, softmax)
#         results[i] = counts

#     #for i in range(len(counts)):
#     #    print str(counts[i]) + " -- " + str(explores[i])

#     average = np.mean(results, axis=0)
#     avgRMS = np.mean(allRMSs, axis=0)

#     return average, avgRMS

# # parameters
# samples = 100
# episodes = 1000
# discount = 0.7
# epsilon = 0.1
# alpha = 0.3
# initValue = 15
# softmax = False
# skip = False
# category = 'discount'

# game = pg.PredatorGame([(0,0), (10,10)], (5,5), (11,11))

# if not skip:


#     results = dict()
#     if category == 'epsilon':
#         for epsilon in [0.05,0.1,0.3,0.9]:
#             results[epsilon], avgRMS = getResults(samples, episodes, discount, epsilon, alpha, initValue, False)
#     elif category == 'tau':
#         for tau in [0.05,12,30,100]:
#             results[tau], avgRMS = getResults(samples, episodes, discount, epsilon, alpha, initValue, True)
#     elif category == 'alpha':
#         for alpha in [0.1,0.2,0.3,0.6,1]:
#             results[alpha], avgRMS = getResults(samples, episodes, discount, epsilon, alpha, initValue, softmax)
#     elif category == 'discount':
#         for discount in [0.1,0.4,0.7,0.8,0.9]:
#             print(discount)
#             results[discount], avgRMS = getResults(samples, episodes, discount, epsilon, alpha, initValue, softmax)
#     else:
#         sys.exit()
#     print(results)
#     results['episode'] = range(0,episodes)
#     dataF = DataFrame(results)
#     dataF.to_pickle('data/'+category+str(softmax))
# else:
#     dataF = pd.read_pickle('data/'+category+str(softmax))

# episodeData = pd.melt(dataF, id_vars=['episode'], var_name=category)


# p = ggplot(episodeData, aes('episode', 'value', color=category)) +\
#     geom_line() +\
#     theme_bw() + theme() + ylab("Steps") + xlab("Episodes") + ylim(0,60)
# print(p)
# ggsave(p, "plots/"+category+str(softmax)+".png")
# ggsave(p, "plots/"+category+str(softmax)+".pdf")
