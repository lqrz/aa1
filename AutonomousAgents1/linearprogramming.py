## ==================


## Guide to setting up linear programming for python


## ==================


## easy_install pulp


## Any problems, then Google ^_^ or https://code.google.com/p/pulp-or/


## ==================


## Example script

import pulp

actions = [0,1]
states = [0,1,2,3,4]

state = 1

Q =pulp.LpVariable.dicts("Q",(states,actions,actions),0,4,pulp.LpInteger)

policy = pulp.LpVariable.dicts("policy",(states,actions),0,4,pulp.LpInteger)

print policy
prob = pulp.LpProblem("myProblem", pulp.LpMinimize)

# Constraint
prob += min([pulp.lpSum([policy[state][action] * Q[state][action][actionPrey] for action in actions]) for actionPrey in actions]) >= Q

# Objective
# prob += -4*x + y

status = prob.solve(pulp.PULP_CBC_CMD(msg = 0))

print pulp.value(policy[state][0])
print pulp.value(policy[state][1])

print pulp.value(Q[state][0][0])
print pulp.value(Q[state][0][1])
print pulp.value(Q[state][1][0])
print pulp.value(Q[state][1][1])