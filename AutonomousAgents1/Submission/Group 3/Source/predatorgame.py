#! /bin/env python

import numpy as np
import time
from matplotlib.table import Table
import matplotlib.pyplot as plt

'''
Predator prey game class -- The goal of the game for the predator is to catch the prey. The prey does not really
have a goal as it exhibits random behaviour. This class supports multiple methods of computing the optimal policy.
'''

class PredatorGame():

        def __init__(self, predOrigin, preyOrigin, boardSize):
                self.predCoord = predOrigin
                self.preyCoord = preyOrigin
                self.boardSize = boardSize
                self.state = 0
                self.moves = [(0,0), (-1,0), (1,0), (0,-1), (0,1)]
                # Moves for the predator, they are inverted from the prey because of the way the state works
                self.mPred = [(0,0), (1,0), (-1,0), (0,1), (0,-1)]
                self.preyProbs = [0.8, 0.05, 0.05, 0.05, 0.05]
                self.preyCumProbs = np.cumsum(self.preyProbs)
                self.predProbs = [0.2, 0.2, 0.2, 0.2, 0.2]
                self.predCumProbs = np.cumsum(self.predProbs)


        # Get the possible following states
        def possibleFollowingStates(self, state, mPred):
            possibleStates = []
            # Update the state by performing the predator move
            state = self.makeMove(state, mPred)
            for i, mPrey in enumerate(self.moves):
                # Update the state by performing the prey move
                newState = self.makeMove(state, mPrey)
                possibleStates.append((newState, self.preyProbs[i]))
            return possibleStates

        # Returns a reduced state spaces based on the distance between the predator and the prey.
        def getStates(self):
            return set([self.getClosestDistance(x-a,y-b) for x in range(self.boardSize[0]) for y in range(self.boardSize[1]) for a in range(self.boardSize[0]) for b in range(self.boardSize[1])])

        # Converts coordinates x, y to state space distances. The state is the distance to the prey,
        # the distance may turn from positive to negative if the fastest route is by going around the
        # board.
        def getClosestDistance(self,x,y):
            return ((x+self.boardSize[0]/2) % self.boardSize[0] - self.boardSize[0]/2, (y+self.boardSize[1]/2) % self.boardSize[1] - self.boardSize[1]/2)
        # Converts coordinates to coordinates restricted to the board size
        def boardCoordinates(self,coords):
            return (coords[0] % self.boardSize[0], coords[1] % self.boardSize[1])
        # Returns the state associated with a combination of predator and prey coordinates
        def getState(self, predCoord, preyCoord):
            return self.getClosestDistance(predCoord[0]-preyCoord[0],predCoord[1]-preyCoord[1])
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
                    for move in self.mPred:
                        # find possible next states
                        nextStates = self.possibleFollowingStates(state, move)

                        tmp = 0
                        for nextState in nextStates:
                            transProb = nextState[1]
                            nextState = nextState[0]
                            reward = 0
                            # Give reward for the end state
                            if nextState == (0,0):
                                reward = 10
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
                            reward = 0
                            # Give reward for the end state
                            if nextState == (0,0):
                                reward = 10
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

        # Returns the maximizing action using the value function and the current predator and prey locations
        def maximizingAction(self, state, discount, V):
            # Evaluates the action value
            maximizingAction = np.argmax([self.actionValue(move, state, discount, V) for move in self.moves])
            # Return the move
            return self.moves[maximizingAction]

        def actionValue(self, move, state, discount, V):
            # find possible next states
            nextStates = self.possibleFollowingStates(state, move)
            tmp = 0
            for state in nextStates:
                transProb = state[1]
                state = state[0]
                reward = 0
                if state == (0,0):
                    reward = 10
                tmp += transProb * (reward + discount * V[state])
            return tmp



        # Choses a random move, and returns the direction of the move.
        # probs: is the accumelated probability
        def choseMovement(self, moves, probs):

            # Choose a random random number
            move = np.random.rand()

            for i in range(len(probs)):
                if move < probs[i]:
                    chosen_index = i
                    break

            return [moves[chosen_index][0], moves[chosen_index][1]]


        # Calculate the new coordinates
        def makeMove(self, origin, direction):
            return self.getClosestDistance(origin[0] + direction[0], origin[1] + direction[1])

        # Generates a random move
        def move(self, origin, moves, probs):
            # Get the random direction of the move
            direction = self.choseMovement(moves, probs)
            # Return the new coordinates
            return self.makeMove(origin, direction)

        # Generate a move for the prey
        def prey(self):
            removed = -1

            # Check for moves that cause the prey to reach the predator
            for i in range(len(self.moves)):
                # If a move is found then stop
                if self.makeMove(self.preyCoord,self.moves[i]) == self.predCoord:
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
            # Change the predator coordinates
            self.preyCoord = self.boardCoordinates(self.move(self.preyCoord, self.moves, self.preyCumProbs))

        # Generate a move for the predator
        def predator(self, policy=None):
            if not policy:
                # Perform a random move if there is no policy
                self.predCoord = self.boardCoordinates(self.move(self.predCoord, self.moves, self.predCumProbs))
            else:
                # Perform a move according to the policy
                move = policy[self.getState(self.predCoord, self.preyCoord)]
                self.predCoord = self.boardCoordinates(self.makeMove(self.predCoord, move))

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