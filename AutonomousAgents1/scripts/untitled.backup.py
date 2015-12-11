#! /bin/env python

import numpy as np
import time, math
from matplotlib.table import Table
import matplotlib.pyplot as plt
import random

'''
Predator prey game class -- The goal of the game for the predator is to catch the prey. The prey does not really
have a goal as it exhibits random behaviour. This class supports multiple methods of computing the optimal policy.
'''

class PredatorGame():

        #-------------- Constructor ------------------------------

        def __init__(self, predOrigin, preyOrigin, boardSize):
                self.predCoord = predOrigin
                self.preyCoord = preyOrigin
                self.boardSize = boardSize
                self.state = 0
                self.moves = [(0,0), (-1,0), (1,0), (0,-1), (0,1)]
                self.preyProbs = [0.8, 0.05, 0.05, 0.05, 0.05]
                self.preyCumProbs = np.cumsum(self.preyProbs)
                self.predProbs = [0.2, 0.2, 0.2, 0.2, 0.2]
                self.predCumProbs = np.cumsum(self.predProbs)

        #-------------- DP Planning Algorithms ------------------------------


        # Executes iterative policy evaluation
        def iterativePolicyEvaluation(self, discount, theta):
            states = self.getStates()

            # initialize V arbitrarily
            V = dict()
            for state in states:
                V[state] = 0

            # repeat until ...
            delta = -1
            c = 0
            while(delta >= theta or delta == -1):
                c += 1
                delta = 0
                for state in states:
                    # Store the previous value
                    oldV = V[state]
                    # Set the new value to 0
                    V[state] = 0
                    for move in self.moves:
                        # find possible next states
                        nextStates = self.possibleFollowingStates(state, move)

                        tmp = 0
                        for nextState in nextStates:
                            transProb = nextState[1]
                            nextState = nextState[0]
                            reward = self.getReward(state, nextState)
                            tmp += transProb * (reward + discount * V[nextState])


                        V[state] += 0.2*tmp

                    delta = max(delta, abs(oldV-V[state]))

            return V, c

        # Executes policy iteration
        def policyIteration(self, discount, theta):
            states = self.getStates()

            # Initialize
            V = dict()
            policy = dict()
            for state in states:
                # initialize V arbitrarily
                V[state] = 0
                # Initialize a random policy
                policy[state] = self.choseMovement(self.moves, self.preyCumProbs)

            c = 0
            stable = False
            # repeat until policy has not changed
            while stable == False:
                # Policy evaluation
                delta = -1
                c += 1
                while(delta >= theta or delta == -1):
                    delta = 0
                    for state in states:
                        oldV = V[state]
                        # find possible next states
                        nextStates = self.possibleFollowingStates(state, policy[state])

                        tmp = 0
                        for nextState in nextStates:
                            transProb = nextState[1]
                            nextState = nextState[0]
                            reward = self.getReward(state, nextState)
                            tmp += transProb * (reward + discount * V[nextState])

                        V[state] = tmp
                        delta = max(delta, abs(oldV-V[state]))
                # Policy improvement
                stable = True
                for state in states:
                    b = policy[state]
                    # Improve policy in state
                    policy[state] = self.maximizingAction(state, discount, V)
                    # If policy changed set stable to False
                    if b != policy[state]:
                        stable = False
            return V, c, policy


        # Executes value iteration
        def valueIteration(self, discount, theta):
            states = self.getStates()

            # initialize V arbitrarily
            V = dict()
            for state in states:
                V[state] = 0

            # repeat until convergence
            delta = -1
            c = 0
            while(delta >= theta or delta == -1):
                c += 1
                delta = 0
                for state in states:
                    oldV = V[state]
                    # Get the maximum action value
                    V[state] = max([self.actionValue(move, state, discount, V) for move in self.moves])
                    delta = max(delta, abs(oldV-V[state]))
            # Generate a policy with all the maximizing actions
            policy = {state: self.maximizingAction(state, discount, V) for state in states}

            return V, c, policy        

        #-------------- Monte Carlo Algorithms ------------------------------
        
        # Execultes On-Policy Monte Carlo Control
        def monteCarloControl(self, discount, epsilon, N, initValue):
            # Initialization
            Q = dict()
            Returns = dict()
            probPolicy = dict()
            for state in self.getStates():
                Q[state] = dict()
                Returns[state] = dict()
                probPolicy[state] = dict()
                for action in self.moves:
                    Q[state][action] = initValue
                    Returns[state][action] = []
                    probPolicy[state][action] = 1.0 / len(self.moves) # random policy

            # Repeat forever
            stepCounts = []
            for episode in range(N):
                self.reset()

                # Generate an episode
                episode = self.generateEpisode(probPolicy)
                # episode[0] : current step
                # episode[1] : action
                # episode[2] : next step

                stepCounts.append(len(episode))
                # for e in episode:
                #     print str(e)
                seenPairs = set()
                for i, step in enumerate(episode):
                    state = step[0]
                    action = step[1]
                    # only process fist visit in first-visit MC
                    if not (state, action) in seenPairs:
                        seenPairs.add((state, action))
                        retrn = sum([(discount ** k) * self.getReward(returnPair[0], returnPair[2]) for k, returnPair in enumerate(episode[i:])]) # Return following first occurrence of s,a getReward()
                        Returns[state][action].append(retrn)
                        Q[state][action] = np.mean(Returns[state][action])
                for state in set([p[0] for p in seenPairs]):
                    # Find the maximizing policy
                    optimalAction = self.getGreedyAction(Q[state])
                    for action in self.moves:
                        if optimalAction == action:
                            probPolicy[state][action] = 1 - epsilon + (float(epsilon) / len(self.moves))
                        else:
                            probPolicy[state][action] = float(epsilon) / len(self.moves)

            return Q, stepCounts
        # Execultes Off-Policy Monte Carlo Control
        def offPolicyMonteCarloControl(self, discount, epsilon, N, initValue):
            # Initialization
            Q = dict()
            Numerator = dict() # Numerator
            Denominator = dict() # Denominator of Q(s,a)
            probPolicy = dict()
            policy = dict()
            for state in self.getStates():
                Q[state] = dict()
                Numerator[state] = dict()
                Denominator[state] = dict()
                policy[state] = self.choseMovement(self.moves, self.preyCumProbs) # Arbitrary determnistic policy
                probPolicy[state] = dict()
                for action in self.moves:
                    Q[state][action] = initValue
                    Numerator[state][action] = 0
                    Denominator[state][action] = 0
                    probPolicy[state][action] = 1.0 / len(self.moves) # random policy

            # Repeat forever
            stepCounts = []
            for episode in range(N):
                self.reset()
                # Select a policy? TODO: What does Select a policy mean?
                # Generate an episode
                episode = self.generateEpisode(probPolicy)
                stepCounts.append(len(episode))
                # for e in episode:
                #     print str(e)
                taus = [i for i in range(0,len(episode)) if episode[i][1] != policy[episode[i][0]]]
                # Default value
                tau = 0
                if len(taus) != 0:
                    tau = taus[-1]
                # Loop over each pair s,a appearing in the episode at time tau or later
                for i, step in enumerate(episode[tau:]):
                    state = step[0]
                    action = step[1]
                    # Time of first occurance
                    time = [i for i in range(tau,len(episode)) if step == episode[i]][0]
                    w = np.prod([1/probPolicy[episode[k][0]][episode[k][1]] for k in range(time+1,len(episode)-1)])
                    retrn = sum([(discount ** k) * self.getReward(returnPair[0], returnPair[2]) for k, returnPair in enumerate(episode[time:])])
                    Numerator[state][action] = Numerator[state][action]+w*retrn
                    Denominator[state][action] = Denominator[state][action]+w
                    Q[state][action] = np.divide(Numerator[state][action],Denominator[state][action])
                for state in self.getStates():
                    # Find the maximizing policy
                    policy[state] = self.getGreedyAction(Q[state])

            return Q, stepCounts

        #-------------- Temporal Difference Learning Algorithms ------------------------------

        # Executes SARSA
        def sarsa(self, discount, alpha, epsilon, N, initQ, softmax):

            # initialize Q arbitrarily
            Q = {s: {a: initQ for a in self.moves} for s in self.getStates() }

            # Init Q for 0,0 to zero
            Q[(0,0)] = {a: 0 for a in self.moves}

            optimalPolicy = dict()
            
            stepCounters = []
            for i in range(N):
                self.reset()
                state = self.getState(self.predCoord, self.preyCoord)
                a = self.getAction(epsilon, Q[state], softmax)
                steps = 0
                while(not self.state):
                    # take action a, observe r, s'
                    policy = dict()
                    policy[state] = a
                    self.step(policy=policy)
                    steps += 1
                    nextState = self.getState(self.predCoord, self.preyCoord)
                    reward = self.getReward(state, nextState)
                    nextAction = self.getAction(epsilon, Q[nextState], softmax)
                    Q[state][a] = Q[state][a] + alpha * (reward + (discount * Q[nextState][nextAction]) - Q[state][a])
                    state = nextState
                    a = nextAction
                stepCounters.append(steps)

            for state in self.getStates():
                # Find the maximizing policy
                optimalPolicy[state] = self.getGreedyAction(Q[state])
            self.state = 0
            return Q, stepCounters, optimalPolicy

        # Executes Qlearning
        def Qlearning(self, discount, epsilon, N, alpha, initQ, softmax):

            # initialize Q arbitrarily
            Q = {s: {a: initQ for a in self.moves} for s in self.getStates() }

            # Init Q for 0,0 to zero
            Q[(0,0)] = {a: 0 for a in self.moves}

            optimalPolicy = dict()

            # repeat for each episode
            stepCounters = []
            for episode in range(N):
                # reset game
                self.reset()
                # step counter
                step = 0
                explore = 0
                # initialize s
                #self.predCoord = tuple(np.random.random_integers(0, 11, 2))
                #self.preyCord = tuple(np.random.random_integers(0, 11, 2))                
                state = self.getState(self.predCoord, self.preyCoord)
                # repeat for each step of the episode
                while(not self.state):
                    action = self.getAction(epsilon, Q[state], softmax)
                    policy = dict()
                    policy[state] = action
                    self.step(policy=policy)
                    step += 1
                    nextState = self.getState(self.predCoord, self.preyCoord)
                    reward = self.getReward(state, nextState)
                    Q[state][action] = Q[state][action] + alpha * (reward + discount * max([Q[nextState][a] for a in self.moves]) - Q[state][action])
                    state = nextState
                stepCounters.append(step)

            for state in self.getStates():
                # Find the maximizing policy
                optimalPolicy[state] = self.getGreedyAction(Q[state])

            self.state = 0
            return Q, stepCounters, optimalPolicy


        #-------------- Simulating Environment Methods ------------------------------
        # These methods move the agents and influence the state of the game.

        def reset(self, pred=(0,0), prey=(5,5)):
            self.predCoord = pred
            self.preyCoord = prey
            self.state = 0

        def generateEpisode(self, probPolicy):
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
        def step(self, policy=None):
            # Perform moves
            self.predator(policy)
            self.prey()
            # Check if in an end state
            if self.predCoord == self.preyCoord:
                    self.state = 1
            # Returns the coordinates of the predator and prey
            # These coordinates are only used for visualization not for the state of the policy.
            return ((self.predCoord, self.preyCoord), self.state)

        # Generate a move for the prey
        def prey(self):
            removed = -1

            # Check for moves that cause the prey to reach the predator
            for i in range(len(self.moves)):
                # If a move is found then stop
                if self.makeMoveOnCoordinates(self.preyCoord,self.moves[i]) == self.predCoord:
                    removed = i
                    break

            # If there is a move which is removed then modify the probability distribution
            if removed != -1:
                for i in range(1,len(self.preyCumProbs)):
                    if i == removed:
                        # Ignore this case by making the probability width 0
                        self.preyCumProbs[i] = self.preyCumProbs[i-1]
                    else:
                        # Modifty the probability of the other cases
                        self.preyCumProbs[i] = self.preyCumProbs[i-1] + (1.0 - self.preyCumProbs[0])/(len(self.preyCumProbs) - 2)
            # Change the prey coordinates
            self.preyCoord = self.boardCoordinates(self.move(self.preyCoord, self.moves, self.preyCumProbs))

        # Generate a move for the predator
        def predator(self, policy=None):
            if not policy:
                # Perform a random move if there is no policy
                self.predCoord = self.boardCoordinates(self.move(self.predCoord, self.moves, self.predCumProbs))
            else:
                # Perform the move according to the policy
                move = policy[self.getState(self.predCoord, self.preyCoord)]
                self.predCoord = self.makeMoveOnCoordinates(self.predCoord, move)


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
            if state != (0,0) and nextState == (0,0):
                reward = 10
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
            return set([self.getClosestDistance(x-a,y-b) for x in range(self.boardSize[0]) for y in range(self.boardSize[1]) for a in range(self.boardSize[0]) for b in range(self.boardSize[1])])


        # Returns the state associated with a combination of predator and prey coordinates
        def getState(self, predCoord, preyCoord):
            return self.getClosestDistance(predCoord[0]-preyCoord[0],predCoord[1]-preyCoord[1])

        #-------------- Auxiliary Methods for Policy / Action Selection ------------------------------
        # These methods do not update the current state of the game. 


        # Returns the maximizing action for Q(s,a) given a state s
        def getGreedyAction(self, Qs):
            moves = list(self.moves)
            random.shuffle(moves)
            a_index = np.argmax([Qs[a] for a in moves])
            return moves[a_index]

        def getAction(self, epsilon, Qs, softmax):
            if softmax:
                action = self.getSoftmaxAction(epsilon, Qs)
            else:
                action = self.getActionEpsilonGreedy(epsilon, Qs)
            return action

        # Returns the softmax action
        def getSoftmaxAction(self, tau, Qs):
            summed = sum([math.exp(Qs[a]/tau) for a in self.moves])
            probs = [math.exp(Qs[a]/tau)/summed for a in self.moves]
            return self.choseMovement(self.moves, np.cumsum(probs))

        # Returns an action according to an epsilon-greedy policy and Q(s).
        # Also returns 1 if the agent was exploring, or 0 if the agent was exploiting.
        def getActionEpsilonGreedy(self, epsilon, Qs):
            "Returns epsilon-greedy action Q"
            i = np.random.rand()
            if i < epsilon:
                # explore
                a = self.choseMovement(self.moves, np.cumsum([float(1)/len(self.moves)]*len(self.moves)))
                return a
            else:
                # exploit
                return self.getGreedyAction(Qs)

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

            return (moves[chosen_index][0], moves[chosen_index][1])


        # Generates a random move for given probabilities and returns the new coordinates (irregardless if they're valid)
        def move(self, origin, moves, probs):
            # Get the random direction of the move
            direction = self.choseMovement(moves, probs)
            # Return the new coordinates
            return self.makeMove(origin, direction)


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
        game = PredatorGame((0,0), (5,5), (11,11))

        print "If no output shows, see example usage at the end of the file."

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
        # preds = [(a, b) for a in range(11) for b in range(11)]
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
        # preds = [(a, b) for a in range(11) for b in range(11)]
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