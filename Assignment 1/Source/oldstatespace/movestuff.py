import numpy as np
import time
from matplotlib.table import Table
import matplotlib.pyplot as plt

class PredatorGame():

        def __init__(self, predOrigin, preyOrigin, boardSize):
                self.predCoord = predOrigin
                self.preyCoord = preyOrigin
                self.boardSize = boardSize
                self.state = 0
                self.moves = [[0,0], [-1,0], [1,0], [0,-1], [0,1]]
                self.preyProbs = [0.8, 0.05, 0.05, 0.05, 0.05]
                self.preyCumProbs = np.cumsum(self.preyProbs)
                self.predProbs = [0.2, 0.2, 0.2, 0.2, 0.2]
                self.predCumProbs = np.cumsum(self.predProbs)


        # Get the possible following states
        def possibleFollowingStates(self, pred, prey, mPred):
            possibleStates = []
            newPred = tuple(self.makeMove(pred, mPred))
            for i, mPrey in enumerate(self.moves):
                newPrey = tuple(self.makeMove(prey, mPrey))
                possibleStates.append((newPred, newPrey, self.preyProbs[i]))

            return possibleStates

       # Executes iterative policy evaluation
        def iterativePolicyEvaluation(self, discount, theta):
            states = [((x, y), (a,b)) for x in range(self.boardSize[0]) for y in range(self.boardSize[1]) for a in range(self.boardSize[0]) for b in range(self.boardSize[1])]

            # initialize V arbitrarily
            V = dict()
            for s in states:
                V[s] = 0

            # repeat until ...
            delta = -1
            c = 0
            while(delta >= theta or delta == -1):
                c += 1
                delta = 0
                for s in states:
                    # Store the previous value
                    v = V[s]

                    # Set the new value to 0
                    V[s] = 0
                    pred = s[0]
                    prey = s[1]
                    for m in self.moves:
                        # find possible next states
                        nextStates = self.possibleFollowingStates(pred, prey, m)

                        tmp = 0
                        for s2 in nextStates:
                            transProb = s2[2]
                            s2 = s2[:2]
                            r = 0
                            if s2[0] == s2[1]:
                                r = 10
                            tmp += transProb * (r + discount * V[s2])


                        V[s] += 0.2*tmp

                    delta = max(delta, abs(v-V[s]))

            return V, c

       # Executes policy iteration
        def policyIteration(self, discount, theta):
            states = [((x, y), (a,b)) for x in range(self.boardSize[0]) for y in range(self.boardSize[1]) for a in range(self.boardSize[0]) for b in range(self.boardSize[1])]

            # initialize V arbitrarily
            V = dict()
            policy = dict()
            for s in states:
                V[s] = 0
                policy[s] = self.choseMovement(self.moves, self.preyCumProbs)

            # repeat until ...
            stable = False
            c = 0
            while stable == False:
                # Policy evaluation
                delta = -1
                c += 1
                while(delta >= theta or delta == -1):
                    delta = 0
                    for s in states:
                        v = V[s]
                        pred = s[0]
                        prey = s[1]

                        # find possible next states
                        nextStates = self.possibleFollowingStates(pred, prey, policy[s])

                        tmp = 0
                        for nextState in nextStates:
                            transProb = nextState[2]
                            nextState = nextState[:2]
                            r = 0
                            if nextState[0] == nextState[1]:
                                r = 10
                            tmp += transProb * (r + discount * V[nextState])

                        V[s] = tmp
                        delta = max(delta, abs(v-V[s]))
                # Policy improvement
                stable = True
                for state in states:
                    b = policy[state]
                    # Improve policy in state
                    policy[state] = self.maximizingAction(state[0], state[1], discount, V)

                    if b != policy[state]:
                        stable = False
            return V, c, policy


        # Executes value iteration
        def valueIteration(self, discount, theta):
            states = [((x, y), (a,b)) for x in range(self.boardSize[0]) for y in range(self.boardSize[1]) for a in range(self.boardSize[0]) for b in range(self.boardSize[1])]

            # initialize V arbitrarily
            V = dict()
            for s in states:
                V[s] = 0

            # repeat until ...
            delta = -1
            c = 0
            while(delta >= theta or delta == -1):
                c += 1
                delta = 0
                for s in states:
                    v = V[s]
                    pred = s[0]
                    prey = s[1]
                    V[s] = max([self.actionValue(move, pred, prey, discount, V) for move in self.moves])
                    delta = max(delta, abs(v-V[s]))

            return V, c

        def maximizingAction(self, pred, prey, discount, V):
            # Evaluates the action value
            maximizingAction = np.argmax([self.actionValue(move, pred, prey, discount, V) for move in self.moves])
            # Return the move
            return self.moves[maximizingAction]

        def actionValue(self, move, pred, prey, discount, V):
            # find possible next states
            nextStates = self.possibleFollowingStates(pred, prey, move)

            tmp = 0
            for s2 in nextStates:
                transProb = s2[2]
                s2 = s2[:2]
                r = 0
                if s2[0] == s2[1]:
                    r = 10
                tmp += transProb * (r + discount * V[s2])
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
            return [(origin[0] + direction[0]) % self.boardSize[0], (origin[1] + direction[1]) % self.boardSize[1]]

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
            self.preyCoord = self.move(self.preyCoord, self.moves, self.preyCumProbs)

        # Generate a move for the predator
        def predator(self):
            self.predCoord = self.move(self.predCoord, self.moves, self.predCumProbs)

        # Take a step in the program
        def step(self):
                self.predator()
                self.prey()
                if self.predCoord == self.preyCoord:
                        self.state = 1
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

        g = PredatorGame([0,0], [5,5], [11,11])
        # c = 0
        # while (not g.state):
        #         c += 1
        #         g.step()
        #         print(str(g.predCoord) + " -- " + str(g.preyCoord))

        # print(c)        # c = 0

        #---------begin call value iteration---------#
        # discount = 0.1
        # threshold = 0.00001
        # preyPosition = tuple((5,5))

        # V, c = g.valueIteration(discount, threshold)

        # example = []
        # preds = [(a, b) for a in range(11) for b in range(11)]
        # for p in preds:
        #     example.append((p, preyPosition))

        # #for e in example:
        # #    print (str(e[0]) + "-->" +  str(round(V[e], 3)))

        # valuesMatrix = np.zeros((g.boardSize[0], g.boardSize[1]))

        # for e in example:
        #     valuesMatrix[e[0][0]][e[0][1]] = round(V[e], 4)



        # g.checkerboard_table(valuesMatrix, preyPosition)
        # plt.show()
        #---------end call value iteration---------#
        #---------begin call policy iteration---------#
        discount = 0.1
        threshold = 0.00001
        preyPosition = tuple((5,5))

        results = []
        for i in range(10):
            g = PredatorGame((0,0), (5,5), (11,11))
            V, c, policy = g.policyIteration(discount, threshold)
            results.append(c) 
            print i
            print results
            print("Mean:" + str(np.mean(results)))
        #policy = np.array(policy)
        #print policy[policy.keys()[1:20]]
        example = []
        preds = [(a, b) for a in range(11) for b in range(11)]
        for p in preds:
            example.append((p, preyPosition))

        #for e in example:
        #    print (str(e[0]) + "-->" +  str(round(V[e], 3)))

        valuesMatrix = np.zeros((g.boardSize[0], g.boardSize[1]))

        for e in example:
            valuesMatrix[e[0][0]][e[0][1]] = round(V[e], 4)


        g.checkerboard_table(valuesMatrix, preyPosition)
        plt.show()
        #---------end call policy iteration---------#


        elapsed_time = time.time() - start_time
        print("Elapsed time: " + str(elapsed_time))
        print(c)


