#! /bin/env python

from Tkinter import *
import time
from predatorgame import PredatorGame

'''
Class to visualize the predator prey agent game.
'''

class Board:


	def __init__(self, offsetX, offsetY, canvasWidth, canvasHeight, N):
		self.master = Tk()
		self.offsetX = offsetX
		self.offsetY = offsetY
		self.canvasWidth = canvasWidth
		self.canvasHeight = canvasHeight
		self.width = self.canvasWidth - self.offsetX*2
		self.height = self.canvasHeight - self.offsetY*2
		self.N = N
		self.canvas = Canvas(self.master, width=self.canvasWidth, height=self.canvasHeight)
		self.drawCheckerBoard()
		self.predators = []
		self.preys = []
		#self.predatorFile = PhotoImage(file = "images/prey.gif")
		#self.preyFile = PhotoImage(file = "images/predator.gif")
		self.g = PredatorGame([(0,0),(10,10),(0,10),(10,0)], (5,5), (11,11))
		self.initAgents([(0,0),(10,10),(0,10),(10,0)],[(5,5)])
		# Draw the intitial state
		self.draw(self.g.predCoords, [self.g.preyCoord])
		# Learn a policy
		self.policy = None
		#self.V, self.c, self.policy = self.g.policyIteration(0.1, 0.00001)
		self.mainloop()
		
		self.canvas.mainloop()

	'''
	Performs a step in the game and draws the canvas (automatically calls itself after 80ms)
	'''
	def mainloop(self):
		if(not self.g.state):
			# Draw a step in the game
			(test, newState), preyMove = self.g.step()
			#print str(self.g.predCoord) + " -- " + str(self.g.preyCoord)
			# Draw the state on the board
			self.draw(self.g.predCoords, [self.g.preyCoord])

			# Check if the game ended
			if newState != 1:
				self.canvas.after(80, self.mainloop)
			else:
				# Restart the game if the game ended
				self.restartGame()

	'''
	Restarts the game
	'''
	def restartGame(self):
		self.g = PredatorGame([(0,0),(10,10),(10,0),(0,10)], (5,5), (11,11)) #TODO: This is honestly a bit of a hack
		self.canvas.after(800, self.mainloop)

	'''
	Draws a checkerboard
	'''
	def drawCheckerBoard(self):
		self.canvas.pack()
		for x in range(0,self.N):
			for y in range(0,self.N):
				# Draws the checkers
				fillColor = 'white' if ((x + y*self.N) % 2 == 0) else "blue";
				self.canvas.create_rectangle(self.offsetX+x*self.width/self.N, self.offsetY+y*self.height/self.N, self.offsetX+(x+1)*self.width/self.N, self.offsetY+(y+1)*self.height/self.N, fill=fillColor)

	'''
	Draws the game screen
	'''
	def draw(self, predatorsCoords, preysCoords):
		# Predators
		for i in range(0,len(predatorsCoords)):
			self.movePredator(self.predators[i], predatorsCoords[i][0], predatorsCoords[i][1])
		# Preys
		for i in range(0,len(preysCoords)):
			self.movePrey(self.preys[i], preysCoords[i][0], preysCoords[i][1])

	'''
	Initialization of the predator and prey sprites
	'''
	def initAgents(self, predatorsCoords, preysCoords):
		# Predators
		for i in range(0,len(predatorsCoords)):
			self.drawPredator(predatorsCoords[i][0], predatorsCoords[i][1])
		# Preys
		for i in range(0,len(preysCoords)):
			self.drawPrey(preysCoords[i][0], preysCoords[i][1])

	'''
	Function to draw the predator
	'''
	def drawPredator(self, x, y):
		#self.predators.append(self.canvas.create_image(self.offsetX+x*self.width/self.N, self.offsetY+y*self.height/self.N, image=self.predatorFile))
		self.predators.append(self.canvas.create_rectangle(self.offsetX+x*self.width/self.N, self.offsetY+y*self.height/self.N, self.offsetX+(x+1)*self.width/self.N, self.offsetY+(y+1)*self.height/self.N, fill="red"))

	'''
	Function to draw the prey
	'''
	def drawPrey(self, x, y):
		#self.preys.append(self.canvas.create_image(self.offsetX+x*self.width/self.N, self.offsetY+y*self.height/self.N, image=self.preyFile))
		self.preys.append(self.canvas.create_rectangle(self.offsetX+x*self.width/self.N, self.offsetY+y*self.height/self.N, self.offsetX+(x+1)*self.width/self.N, self.offsetY+(y+1)*self.height/self.N, fill="green"))

	'''
	Function to move the predator sprite
	'''
	def movePredator(self, predator, x, y):
		#self.canvas.coords(predator, self.offsetX+x*self.width/self.N, self.offsetX+y*self.height/self.N)
		self.canvas.coords(predator, self.offsetX+x*self.width/self.N, self.offsetX+y*self.height/self.N, self.offsetX+(x+1)*self.width/self.N, self.offsetY+(y+1)*self.height/self.N)

	'''
	Function to move the prey sprite
	'''
	def movePrey(self, prey, x, y):
		##self.canvas.coords(prey, self.offsetX+x*self.width/self.N, self.offsetX+y*self.height/self.N)
		self.canvas.coords(prey, self.offsetX+x*self.width/self.N, self.offsetX+y*self.height/self.N, self.offsetX+(x+1)*self.width/self.N, self.offsetY+(y+1)*self.height/self.N)

# Call to initialize the board
Board(10, 10, 500, 500, 11)