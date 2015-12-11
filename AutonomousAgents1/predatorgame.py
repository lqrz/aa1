#! /bin/env python

import numpy as np
import time, math
from matplotlib.table import Table
import matplotlib.pyplot as plt
import random
import sys
#import pulp

'''
Predator prey game class -- The goal of the game for the predator is to catch the prey. The prey does not really
have game goal as it exhibits random behaviour. This class supports multiple methods of computing the optimal policy.
'''

class PredatorGame():

        #-------------- Constructor ------------------------------

        def __init__(self, predOrigins, preyOrigin, boardSize):
                self.predCoords = predOrigins[:]
                self.initPredCoords = list(self.predCoords[:])
                self.preyCoord = preyOrigin[:]
                self.initPreyCoord = self.preyCoord[:]
                self.boardSize = boardSize
                self.state = 0
                self.moves = [(0,0), (-1,0), (1,0), (0,-1), (0,1)]
                self.preyProbs = [0.8, 0.05, 0.05, 0.05, 0.05]
                self.preyCumProbs = np.cumsum(self.preyProbs)
                self.predProbs = [0.2, 0.2, 0.2, 0.2, 0.2]
                self.predCumProbs = np.cumsum(self.predProbs)

        
        # Executes Independent Qlearning
        def IndependentQlearning(self, discount, epsilon, N, alpha, initQ, softmax):

            moves = self.getMoves()
            # initialize Q arbitrarily
            Q = {s: {a: initQ for a in moves} for s in self.getStates() }
            Qprey = {s: {a: initQ for a in self.moves} for s in self.getStates() }

            # Init Q for terminal states to zero
            for state in self.getTerminalStates():
                Q[state] = {a: 0 for a in moves}
                Qprey[state] = {a: 0 for a in self.moves}

            # optimal policy
            optimalPolicy     = dict()
            optimalPolicyPrey = dict()
            
            # Dictionary for storing variables used for analysis
            statsDict = dict()
            
            Vmc = dict()
            states = self.getStates()
            # repeat for each episode
            stepCounters = []
            rewards = []
            for episode in range(N):
                print 'episode: '+str(episode)
                # reset game
                self.reset()
                # step counter
                step = 0
                explore = 0
                # initialize s
                #self.predCoord = tuple(np.random.random_integers(0, 11, 2))
                #self.preyCord = tuple(np.random.random_integers(0, 11, 2))
                # repeat for each step of the episode
                state, predatorOrder = self.getState(self.predCoords, self.preyCoord)
                while(not self.state):
                    policy = dict()
                    preyPolicy = dict()
                    action = self.getAction(self.getMoves(), epsilon, Q[state], softmax)
                    policy[state] = action

                    actionPrey = self.getAction(self.moves, epsilon, Qprey[state], softmax)
                    preyPolicy[state] = actionPrey
                    self.step(policy=policy, preyPolicy=preyPolicy)
                    step += 1
                    nextState, predatorOrder = self.getState(self.predCoords, self.preyCoord)
                    reward = self.getReward(state, nextState)
                    rewardPrey = -reward

                    Q[state][action] = Q[state][action] + alpha * (reward + discount * max([Q[nextState][a] for a in self.getMoves()]) - Q[state][action])
                    Qprey[state][actionPrey] = Qprey[state][actionPrey] + alpha * (rewardPrey + discount * max([Qprey[nextState][a] for a in self.moves]) - Qprey[state][actionPrey])
                    state = nextState
                stepCounters.append(step)
                rewards.append(reward)
                
            statsDict['rewards'] = rewards
                

            for state in self.getStates():
                # Find the maximizing policy
                optimalPolicy[state] = self.getGreedyAction(self.getMoves(),Q[state])
                optimalPolicyPrey[state] = self.getGreedyAction(self.moves,Qprey[state])
            
            
            return Q, Qprey, stepCounters, optimalPolicy, optimalPolicyPrey, statsDict

        # Executes SARSA
        def independentSarsa(self, discount, alpha, epsilon, N, initQ, softmax):

            moves = self.getMoves()
            # initialize Q arbitrarily
            Q = {s: {a: initQ for a in self.getMoves()} for s in self.getStates() }
            Qprey = {s: {a: initQ for a in self.moves} for s in self.getStates() }

            # Init Q for terminal states to zero
            for state in self.getTerminalStates():
                Q[state] = {a: 0 for a in moves}
                Qprey[state] = {a: 0 for a in self.moves}
            
            states = self.getStates()
            
            # optimal policy
            optimalPolicy     = dict()
            optimalPolicyPrey = dict()
            rewards = []
            stepCounters = []
            for i in range(N):
                print 'episode: '+str(i)
                self.reset()
                state, predatorOrder = self.getState(self.predCoords, self.preyCoord)
                a = self.getAction(moves, epsilon, Q[state], softmax)
                aPrey = self.getAction(self.moves, epsilon, Qprey[state], softmax)
                steps = 0
                while(not self.state):
                    # take action a, observe r, s'
                    policy = dict()
                    policyPrey = dict()
                    policy[state] = a
                    policyPrey[state] = aPrey
                    self.step(policy=policy, preyPolicy=policyPrey)
                    steps += 1
                    nextState, predatorOrder = self.getState(self.predCoords, self.preyCoord)
                    reward = self.getReward(state, nextState)
                    rewardPrey = -reward
                    nextAction = self.getAction(moves, epsilon, Q[nextState], softmax)
                    nextActionPrey = self.getAction(self.moves, epsilon, Qprey[state], softmax)
                    Q[state][a] = Q[state][a] + alpha * (reward + (discount * Q[nextState][nextAction]) - Q[state][a])
                    Qprey[state][a] = Qprey[state][aPrey] + alpha * (rewardPrey + (discount * Qprey[nextState][nextActionPrey]) - Qprey[state][aPrey])
                    state = nextState
                    a = nextAction
                    aPrey = nextActionPrey
                stepCounters.append(steps)
                rewards.append(reward)
                
            for state in self.getStates():
                # Find the maximizing policy
                optimalPolicy[state] = self.getGreedyAction(moves, Q[state])
                optimalPolicyPrey[state] = self.getGreedyAction(self.moves, Qprey[state])
            
            return Q, Qprey, stepCounters, optimalPolicy, optimalPolicyPrey, dict({'rewards': rewards})


        # Executes minimaxQ
        def minimaxQ(self, exploreProb, discount, decay, N):
            # initialize
            Q = dict()
            policy = dict()
            V = dict()
            moves = self.getMoves()
            for s in self.getStates():
                Q[s] = dict()
                V[s] = 1.0
                policy[s] = dict()
                for a in moves:
                    Q[s][a] = dict()
                    policy[s][a] = 1.0 / len(moves)
                    for o in self.moves: # opponent action of the prey
                        Q[s][a][o] = 1.0
            V[((0,0),)] = 0.0
            alpha = 1.0

            totalSteps = []
            for episode in range(N):
                print episode
                self.reset()
                steps = 0
                while(not self.state):
                    steps += 1
                    currState, ordering = self.getState(self.predCoords, self.preyCoord)
                    # with probability explore, return an action at random (epsilon-greedy)
                    # maybe wrap in function, but Q-values are now Q(s,a,o)
                    if np.random.rand() < exploreProb:
                        a = self.chooseActionProbPolicy({a: 1.0/len(moves) for a in moves})
                    # otherwise if current state is s
                    else:
                        # return maximizing action
                        a = self.getGreedyAction(self.getMoves(), policy[currState])
                        #a = self.getMoves()[np.argmax([policy[currState][a] for a in moves])]

                    # Learn

                    # remember current prey position to see what its move was later
                    currPreyCoord = self.preyCoord

                    # make actual move a

                    # not sure here yet, I guess we need to construct a dummy policy
                    predTempPolicy = {currState: a}
                    _, o = self.step(policy=predTempPolicy)

                    # observe new state and get reward for currState, a, newState:
                    newState = self.getState(self.predCoords, self.preyCoord)[0]
                    reward = self.getReward(currState, newState)
                    # print "old: " + str(currPreyCoord)
                    # print "new: " + str(self.preyCoord)
                    # o = tuple(np.array(self.preyCoord) - np.array(currPreyCoord))
                    # print "o:  " + str(o)

                    # update Q
                    # print Q[currState][a][o]
                    # print alpha
                    # print reward
                    Q[currState][a][o] = (1.0-alpha) * Q[currState][a][o] + alpha * (reward + discount * V[newState])
                    # print a
                    # print o
                    # print Q[currState][a][o]
                    # Use linear programming to find pi[s,.] such that:

                    probs, Vnew = self.linProg(Q, currState)


                    for i, a in enumerate(self.getMoves()):
                        policy[currState][a] = probs[i]

                    # print "----"
                    # print "state: " + str(currState)
                    # for a in policy[currState]:
                    #     print str(a) + " -> " + str(policy[currState][a])
                    # print

                    

                    V[currState] = Vnew

                    alpha = alpha * decay   

                totalSteps.append(steps)
            return policy, totalSteps, V



        def linProg(self, Q, s):

            prob = pulp.LpProblem("myProblem", pulp.LpMaximize)

            # declare your variables
            x1 = pulp.LpVariable("p1",0,  upBound=1, cat='Continuous')   # 0<= x1 <= 40
            x2 = pulp.LpVariable("p2",0,  upBound=1, cat='Continuous')   # 0<= x1 <= 40
            x3 = pulp.LpVariable("p3",0,  upBound=1, cat='Continuous')   # 0<= x1 <= 40
            x4 = pulp.LpVariable("p4",0,  upBound=1, cat='Continuous')   # 0<= x1 <= 40                
            x5 = pulp.LpVariable("p5",0,  upBound=1, cat='Continuous')   # 0<= x1 <= 40
            V = pulp.LpVariable("V", 0, 10 ) 

            # constraints
            prob += pulp.lpSum([x1, x2, x3, x4, x5]) <= 1

            moves = self.getMoves()
            prob += pulp.lpSum([x1 * Q[s][moves[0]][self.moves[0]], x2 * Q[s][moves[1]][self.moves[0]],  x3 * Q[s][moves[2]][self.moves[0]], x4 * Q[s][moves[3]][self.moves[0]], x5 * Q[s][moves[4]][self.moves[0]]]) >= V
            prob += pulp.lpSum([x1 * Q[s][moves[0]][self.moves[1]], x2 * Q[s][moves[1]][self.moves[1]],  x3 * Q[s][moves[2]][self.moves[1]], x4 * Q[s][moves[3]][self.moves[1]], x5 * Q[s][moves[4]][self.moves[1]]]) >= V
            prob += pulp.lpSum([x1 * Q[s][moves[0]][self.moves[2]], x2 * Q[s][moves[1]][self.moves[2]],  x3 * Q[s][moves[2]][self.moves[2]], x4 * Q[s][moves[3]][self.moves[2]], x5 * Q[s][moves[4]][self.moves[2]]]) >= V
            prob += pulp.lpSum([x1 * Q[s][moves[0]][self.moves[3]], x2 * Q[s][moves[1]][self.moves[3]],  x3 * Q[s][moves[2]][self.moves[3]], x4 * Q[s][moves[3]][self.moves[3]], x5 * Q[s][moves[4]][self.moves[3]]]) >= V
            prob += pulp.lpSum([x1 * Q[s][moves[0]][self.moves[4]], x2 * Q[s][moves[1]][self.moves[4]],  x3 * Q[s][moves[2]][self.moves[4]], x4 * Q[s][moves[3]][self.moves[4]], x5 * Q[s][moves[4]][self.moves[4]]]) >= V

            # print prob.constraints



            # objective
            prob += V     
            # print prob.objective     
            # print prob.constraints
            status = prob.solve(pulp.PULP_CBC_CMD(msg = 0))
            #status = prob.solve(pulp.solvers.GLPK_CMD(msg = 0))            

            # print "Values"
            # print Q[s][moves[1]]
            # for move in self.moves:
            #     print "move"
            #     print move
            #     print sum([pulp.value(x1) * Q[s][moves[0]][move], pulp.value(x2) * Q[s][moves[1]][move],  pulp.value(x3) * Q[s][moves[2]][move], pulp.value(x4) * Q[s][moves[3]][move], pulp.value(x5) * Q[s][moves[4]][move]])
            return [pulp.value(x1), pulp.value(x2), pulp.value(x3), pulp.value(x4), pulp.value(x5)], pulp.value(V)





        #-------------- Simulating Environment Methods ------------------------------
        # These methods move the agents and influence the state of the game.

        # 'initial': set the coordinates to the initial values in the constructor
        # None: randomize the coordinates
        # else: set the coordinates to the values provided in the arguments
        def reset(self, preds='initial', prey='initial'):       
            coords = [(x,y) for x in range(self.boardSize[0]) for y in range(self.boardSize[1])]
            invalidCoordsIndices = []
            
            if preds == 'initial':
                self.predCoords = list(self.initPredCoords[:])
            else:
                if preds is None:
                    for i in range(len(self.initPredCoords)):
                        chosen = np.random.choice([k for k in range(len(coords)) if not k in invalidCoordsIndices])
                        self.predCoords[i] = coords[chosen]
                        invalidCoordsIndices.append(chosen)
                else:
                    self.predCoords = preds
            
            if prey == 'initial':
                self.preyCoord = self.initPreyCoord
            else:
                if prey is None:
                    chosen = np.random.choice([k for k in range(len(coords)) if not k in invalidCoordsIndices])
                    self.preyCoord = coords[chosen]
                    invalidCoordsIndices.append(chosen)
                else:
                    self.preyCoord = prey

            # TODO: Fix this check
            # if self.preyCoord == self.predCoord:
            #     if random.choice([True, False]):
            #         self.preyCoord = ((self.preyCoord[0] +  np.random.randint(1, self.boardSize[0])) % self.boardSize[0], self.preyCoord[1])
            #     else:
            #         self.preyCoord = (self.preyCoord[0], (self.preyCoord[1] +  np.random.randint(1, self.boardSize[1])) % self.boardSize[1])
            self.state = 0

        def generateEpisode(self, probPolicy, predCoord=None, preyCoord=None):
            self.reset(predCoord, preyCoord)
            steps = []
            while(not self.state):
                stateCoord = (self.predCoord, self.preyCoord)
                state = self.getState(*stateCoord)
                policy = dict()
                action = self.chooseActionProbPolicy(probPolicy[state])
                policy[state] = action
                self.step(policy=policy)
                nextStateCoord = (self.predCoord, self.preyCoord)
                steps.append((state, action, self.getState(*nextStateCoord)))
            return steps


        # Take a step in the program, and return the coordiantes of the predator and prey.
        def step(self, policy=None, preyPolicy=None):
            predMoves = [(0,0)]*len(self.predCoords)
            # Decide on moves
            predMoves, predatorOrder = self.predator(policy)
            preyMove = self.prey(preyPolicy)
            # Perform moves
            for predatorId in predatorOrder:
                self.predCoords[predatorId] = self.makeMoveOnCoordinates(self.predCoords[predatorId], predMoves[predatorId])
            if np.random.rand() <= 0.8:
                self.preyCoord = self.makeMoveOnCoordinates(self.preyCoord, preyMove)
            # else:
            #     print("Trip!")
            # Evaluate new state
            if not len(self.predCoords) == len(set(self.predCoords)):
                #reward = -10
                self.state = 1
            elif True in [predCoord == self.preyCoord for predCoord in self.predCoords]:
                #reward = 10
                self.state = 1

            # Returns the coordinates of the predator and prey
            # These coordinates are only used for visualization not for the state of the policy.
            return ((self.predCoords, self.preyCoord), self.state), preyMove

        # Generate a move for the prey
        def prey(self, policy=None):
            if not policy:
                # Perform a random move if there is no policy
                move = self.choseMovement(self.moves, self.preyCumProbs)
            else:
                # Perform the move according to the policy
                state, predatorOrder = self.getState(self.predCoords, self.preyCoord)
                move = policy[state]
                #self.preyCoord = self.makeMoveOnCoordinates(self.preyCoord, move)
            return move

        # Generate a move for the predator
        def predator(self, policy=None):
            if not policy:
                # Perform a random move if there is no policy
                move = self.choseMovement(self.moves, self.predCumProbs)
                predatorOrder = range(len(self.predCoords)) # TODO: Do we need to change this?
            else:
                # Perform the move according to the policy
                state, predatorOrder = self.getState(self.predCoords, self.preyCoord)
                move = policy[state]
            return move, predatorOrder


        #-------------- Environment Control ------------------------------
        # These methods implicitly define the behaviour of the environment.
        # These methods do not update the current state of the game.

        # Calculate the new coordinates
        def makeMoveOnCoordinates(self, origin, direction):
            return self.boardCoordinates((origin[0] + direction[0], origin[1] + direction[1]))
        # Calculate the new state
        def makeMove(self, origin, direction):
            return self.getClosestDistance(origin[0] + direction[0], origin[1] + direction[1])
        
        def getReward(self, state, nextState):
            reward = 0
            # Give reward for the end state
            #lqrz - this was before assignment3 modifications 
            #if state != (0,0) and nextState == (0,0):
            #    reward = 10
            #lqrz - check if any predator bumped into another (this goes before checking having caught the prey)
            tuples = [nextState[i] for i in range(len(self.predCoords))]
            for idx,val in enumerate(tuples):
                del tuples[idx]
                if val in tuples:
                    return -10

            #lqrz - check if any predator caught the prey
            for i in range(len(self.predCoords)):
                if nextState[i] == (0,0):
                    return 10
            return reward

        # Converts coordinates x, y to state space distances. The state is the distance to the prey,
        # the distance may turn from positive to negative if the fastest route is by going around the
        # board.
        def getClosestDistance(self,x,y):
            return ((x+self.boardSize[0]/2) % self.boardSize[0] - self.boardSize[0]/2, (y+self.boardSize[1]/2) % self.boardSize[1] - self.boardSize[1]/2)

        # Converts coordinates to coordinates restricted to the board size
        def boardCoordinates(self,coords):
            return (coords[0] % self.boardSize[0], coords[1] % self.boardSize[1])




        #-------------- Auxiliary Environment Methods for States ------------------------------
        # These methods do not update the current state of the game


        # Get the possible following states
        def possibleFollowingStates(self, oldState, mPred):
            possibleStates = []
            # Update the state by performing the predator move
            state = self.makeMove(oldState, mPred)

            # check for end state: either prey is already caught or predator just caught prey
            if oldState == (0,0) or state == (0,0):
                return [((0,0), 1)] # next state after end state is always end state

            for i, mPrey in enumerate(self.moves):
                # Update the state by performing the prey move
                newState = self.makeMove(state, mPrey)
                prob = self.preyProbs[i]
                if state == (0,0):
                    prob = 1 # TODO: Fix this ugly part
                possibleStates.append((newState, prob))
            return possibleStates

        # Returns a reduced state spaces based on the distance between the predator and the prey.
        def getStates(self):
            possibleDistances = set([self.getClosestDistance(x-a,y-b) for x in range(self.boardSize[0]) for y in range(self.boardSize[1]) for a in range(self.boardSize[0]) for b in range(self.boardSize[1])])

            possibleStates = set([(tuple(possibleDistance),) for possibleDistance in possibleDistances])
            for predatorId in range(len(self.predCoords)-1):
                possibleStates = set([tuple(sorted(tuple(possibleState)+(tuple(possibleDistance),))) for possibleState in possibleStates for possibleDistance in possibleDistances])

            return possibleStates

        def getTerminalStates(self):
            states = self.getStates()
            terminalStates = [state for state in states if (not len(state) == len(set(state))) or (True in [stateTuple == (0,0) for stateTuple in state])]
            return terminalStates

        def getMoves(self):
            moves = [(move,) for move in self.moves]
            for i in range(len(self.predCoords)-1):
                moves = [movesTuple+(move,) for move in self.moves for movesTuple in moves]
            return moves

        # Returns the state associated with a combination of predator and prey coordinates
        def getState(self, predCoords, preyCoord):
            #lqrz - this returns a tuple of a list of tuples. Differs from getState(), so cannot index dict later.
            #return tuple([self.getClosestDistance(predCoords[predatorId][0]-preyCoord[0],predCoords[predatorId][1]-preyCoord[1])]+sorted([self.getClosestDistance(predCoords[i][0]-preyCoord[0],predCoords[i][1]-preyCoord[1]) for i in range(len(predCoords)) if i != predatorId]))
            
            #lqrz - it needs to return one tuple with values.
            stateIndex = sorted([(self.getClosestDistance(predCoords[i][0]-preyCoord[0],predCoords[i][1]-preyCoord[1]), i) for i in range(len(predCoords))])
            state = tuple([x[0] for x in stateIndex])
            predatorOrder = [x[1] for x in stateIndex]
            return state, predatorOrder
        #-------------- Auxiliary Methods for Policy / Action Selection ------------------------------
        # These methods do not update the current state of the game.


        # Returns the maximizing action for Q(s,a) given a state s
        def getGreedyAction(self, moves, Qs):
            movesTemp = list(moves)
            random.shuffle(movesTemp)
            a_index = np.argmax([Qs[a] for a in movesTemp])
            return movesTemp[a_index]

        def getAction(self, moves, epsilon, Qs, softmax):
            if softmax:
                action = self.getSoftmaxAction(moves, epsilon, Qs)
            else:
                action = self.getActionEpsilonGreedy(moves, epsilon, Qs)
            return action

        # Returns the softmax action
        def getSoftmaxAction(self, moves, tau, Qs):
            summed = sum([math.exp(Qs[a]/tau) for a in moves])
            probs = [math.exp(Qs[a]/tau)/summed for a in moves]
            return self.choseMovement(moves, np.cumsum(probs))

        # Returns an action according to an epsilon-greedy policy and Q(s).
        # Also returns 1 if the agent was exploring, or 0 if the agent was exploiting.
        def getActionEpsilonGreedy(self, moves, epsilon, Qs):
            "Returns epsilon-greedy action Q"
            i = np.random.rand()
            if i < epsilon:
                # explore
                a = self.choseMovement(moves, np.cumsum([float(1)/len(moves)]*len(moves)))
                return a
            else:
                # exploit
                return self.getGreedyAction(moves, Qs)

        # Chooses an action according to a probabilistic policy p(s)
        def chooseActionProbPolicy(self, actionProbs):
            actions = actionProbs.keys()
            cumProbs = np.cumsum([actionProbs[a] for a in actions])
            action = self.choseMovement(actions, cumProbs)
            return action

        # Returns the maximizing action for a given state using the Value function --> V(s)
        def maximizingAction(self, state, discount, V):
            # Evaluates the action value
            maximizingAction = np.argmax([self.actionValue(move, state, discount, V) for move in self.moves])
            # Return the move
            return self.moves[maximizingAction]
        
        # Same as function maximizingAction() but this function can return more than one actions
        def getAllMaximizingActions(self, state, discount, V):
            maximizingActions = [self.maximizingAction(state, discount, V)]
            
            if not ((state[0] == 0 and maximizingActions[0][0] == 0) or (state[1] == 0 and maximizingActions[0][1] == 0)):
                if maximizingActions[0][0] != 0:
                    if(state[1] > 0):
                        maximizingActions.append((0,-1))
                    else:
                        maximizingActions.append((0,1))
                else:
                    if(state[0] > 0):
                        maximizingActions.append((-1,0))
                    else:
                        maximizingActions.append((1,0))
                            
            # Return the moves
            return maximizingActions

        # Calculates the partial/temporary value for a given state, action pair (aux. method)
        def actionValue(self, move, state, discount, V):
            # find possible next states
            nextStates = self.possibleFollowingStates(state, move)
            tmp = 0
            for nextState in nextStates:
                transProb = nextState[1]
                nextState = nextState[0]
                reward = self.getReward(state, nextState)
                tmp += transProb * (reward + discount * V[nextState])
            return tmp



        # Choses a random move from given action probabilities, and returns the direction of the move.
        # probs: is the accumelated probability
        def choseMovement(self, moves, probs):

            # Choose a random random number
            move = np.random.rand()
            if len(moves) != len(probs):
                probs = [1] # TODO: Fix this
            for i in range(len(probs)):
                if move < probs[i]:
                    chosen_index = i
                    break
            return moves[chosen_index]



        # Generates a random move for given probabilities and returns the new coordinates (irregardless if they're valid)
        def move(self, origin, moves, probs):
            # Get the random direction of the move
            direction = self.choseMovement(moves, probs)
            # Return the new coordinates
            return self.makeMoveOnCoordinates(origin, direction)


        #-------------- Auxiliary Methods Visualization ------------------------------
        # These methods do not update the current state of the game.

        # Displays data (usually V) on the grid for a fixed prey position.
        def checkerboard_table(self, data, preyPosition):
            fig, ax = plt.subplots()
            ax.set_axis_off()
            tb = Table(ax, bbox=[0,0,1,1])

            nrows, ncols = data.shape
            width, height = 1.0 / ncols, 1.0 / nrows

            # Add cells
            for i in range(self.boardSize[0]):
                for j in range(self.boardSize[1]):
                    if(i == preyPosition[0] and j == preyPosition[1]):
                        tb.add_cell(i, j, width, height, text=str(data[i][j]),
                            loc='center', facecolor='yellow')
                    else:
                        tb.add_cell(i, j, width, height, text=str(data[i][j]),
                            loc='center', facecolor='white')

            # Row Labels...
            for i in range(self.boardSize[0]):
                tb.add_cell(i, -1, width, height, text=str(i), loc='right',
                            edgecolor='none', facecolor='none')
            # Column Labels...
            for j in range(self.boardSize[1]):
                tb.add_cell(-1, j, width, height/2, text=str(j), loc='center',
                                   edgecolor='none', facecolor='none')
            ax.add_table(tb)
            #return fig


if "__main__" == __name__:
        start_time = time.time()
        # initialize new game with predator at position 0,0 and prey at position 5,5 on a 11x11 grid
        #game = PredatorGame((0,0), (5,5), (11,11))

        print("If no output shows, see example usage at the end of the file.")

        #---------Example: Random Policy ---------#
        # print "\n----------------\nExample: Random Policy\n----------------"
        # results = []
        # for i in range(100):
        #     game = PredatorGame((0,0), (5,5), (11,11))
        #     c = 0
        #     while (not game.state):
        #              c += 1
        #              game.step()
        #              #print(str(game.predCoord) + " -- " + str(game.preyCoord))

        #     results.append(c)        # c = 0

        # mean = np.mean(results)
        # stdv = np.std(results)
        # print "Min: " + str(min(results))
        # print "Max: " + str(max(results))
        # print "Mean: " + str(mean)
        # print "Standard deviation: " + str(stdv)
        #---------end Task 1---------------------#


        #---------Example: Value Iteration ---------#
        # print "\n----------------\nExample: Value Iteration\n----------------"
        # discount = 0.1
        # threshold = 0.00001
        # preyPosition = (5,5)

        # V, c, policy = game.valueIteration(discount, threshold)

        # example = []

        # #
        # preds = [(game, b) for game in range(11) for b in range(11)]
        # for p in preds:
        #     example.append((p, preyPosition))

        # #for e in example:
        # #    print (str(e[0]) + "-->" +  str(round(V[e], 3)))

        # valuesMatrix = np.zeros((game.boardSize[0], game.boardSize[1]))

        # for e in example:
        #     print e
        #     valuesMatrix[e[0][0]][e[0][1]] = round(V[game.getState(*e)], 4)


        # game.checkerboard_table(valuesMatrix, preyPosition)
        # plt.show()
        #---------end Value Iteration ----------------#


        #---------Example: Policy Iteration --------#
        # print "\n----------------\nExample: Policy Iteration\n----------------"
        # discount = 0.9
        # threshold = 0.00001
        # preyPosition = (5,5)

        # V, c, policy = game.policyIteration(discount, threshold)

        # example = []
        # preds = [(game, b) for game in range(11) for b in range(11)]
        # for p in preds:
        #     example.append((p, preyPosition))

        # #for e in example:
        # #    print (str(e[0]) + "-->" +  str(round(V[e], 3)))

        # valuesMatrix = np.zeros((game.boardSize[0], game.boardSize[1]))

        # for e in example:
        #     valuesMatrix[e[0][0]][e[0][1]] = round(V[game.getState(*e)], 4)

        # game.checkerboard_table(valuesMatrix, preyPosition)
        # plt.show()
        #---------end Policy Iteration---------#


        #---------Example Iterative Policy Evaluation --------#
        # print "\n----------------\nExample: Iterative Policy Evaluation\n----------------"
        # discount = 0.8
        # threshold = 0.00001

        # V, c = game.iterativePolicyEvaluation(discount, threshold)

        # targets = [((0,0), (5,5)), ((2,3), (5,4)), ((2,10), (10,0)), ((10,10), (0,0))]

        # for ex in targets:
        #     print str(ex) +  ": "  + str(V[game.getState(*ex)])
        # print "Iterations: " + str(c)

        #---------end call iterative policy evaluation---------#



        elapsed_time = time.time() - start_time
        print("Elapsed time: " + str(elapsed_time))