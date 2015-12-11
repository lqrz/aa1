import sys
import numpy as np
import pandas as pd
from pandas import DataFrame
import json

sys.path.append("..")
sys.path.append("./")
from ggplot import *
import predatorgame as pg

f = open('scripts/q-learning fixed init 5 samples.txt', 'r')
fixedinit = json.load(f)
f.close()

f = open('scripts/q-learning random init 5 samples.txt', 'r')
randominit = json.load(f)
f.close()

print fixedinit
print randominit

results = dict()
results['Fixed'] = fixedinit
results['Random'] = randominit
results['episode'] = range(0,len(fixedinit))
dataF = DataFrame(results)

episodeData = pd.melt(dataF, id_vars=['episode'], var_name='Starting position')

p = ggplot(episodeData, aes('episode', 'value', color='Starting position')) +\
    geom_line() +\
    theme_bw() + theme() + ylab("RMSE") + xlab("Episodes") + ylim(0,2)
print p
ggsave(p, "plots/startingPositions.png")
ggsave(p, "plots/startingPositions.pdf")
