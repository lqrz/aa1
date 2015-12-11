#! /bin/env python

from Tkinter import *
import time
from movestuff import PredatorGame

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
		self.g = PredatorGame([0,0], [5,5], [11,11])
		self.initAgents([[4,4]],[[5,4]])
		self.mainloop()
		
		self.canvas.mainloop()

	def mainloop(self):

		if(not self.g.state):
			self.g.step()
			print str(self.g.predCoord) + " -- " + str(self.g.preyCoord)
			self.draw([self.g.predCoord], [self.g.preyCoord])

		self.canvas.after(80, self.mainloop)




	def drawCheckerBoard(self):
		self.canvas.pack()
		for x in range(0,self.N):
			for y in range(0,self.N):
				# Draws the checkers
				fillColor = 'white' if ((x + y*self.N) % 2 == 0) else "blue";
				self.canvas.create_rectangle(self.offsetX+x*self.width/self.N, self.offsetY+y*self.height/self.N, self.offsetX+(x+1)*self.width/self.N, self.offsetY+(y+1)*self.height/self.N, fill=fillColor)

	def draw(self, predatorsCoords, preysCoords):
		# Predators
		for i in range(0,len(predatorsCoords)):
			self.movePredator(self.predators[i], predatorsCoords[i][0], predatorsCoords[i][1])
		# Preys
		for i in range(0,len(preysCoords)):
			self.movePrey(self.preys[i], preysCoords[i][0], preysCoords[i][1])

	def initAgents(self, predatorsCoords, preysCoords):
		# Predators
		for i in range(0,len(predatorsCoords)):
			self.drawPredator(predatorsCoords[i][0], predatorsCoords[i][1])
		# Preys
		for i in range(0,len(preysCoords)):
			self.drawPrey(preysCoords[i][0], preysCoords[i][1])

	def drawPredator(self, x, y):
		#self.predators.append(self.canvas.create_image(self.offsetX+x*self.width/self.N, self.offsetY+y*self.height/self.N, image=self.predatorFile))
		self.predators.append(self.canvas.create_rectangle(self.offsetX+x*self.width/self.N, self.offsetY+y*self.height/self.N, self.offsetX+(x+1)*self.width/self.N, self.offsetY+(y+1)*self.height/self.N, fill="red"))

	def drawPrey(self, x, y):
		#self.preys.append(self.canvas.create_image(self.offsetX+x*self.width/self.N, self.offsetY+y*self.height/self.N, image=self.preyFile))
		self.preys.append(self.canvas.create_rectangle(self.offsetX+x*self.width/self.N, self.offsetY+y*self.height/self.N, self.offsetX+(x+1)*self.width/self.N, self.offsetY+(y+1)*self.height/self.N, fill="green"))

	def movePredator(self, predator, x, y):
		#self.canvas.coords(predator, self.offsetX+x*self.width/self.N, self.offsetX+y*self.height/self.N)
		self.canvas.coords(predator, self.offsetX+x*self.width/self.N, self.offsetX+y*self.height/self.N, self.offsetX+(x+1)*self.width/self.N, self.offsetY+(y+1)*self.height/self.N)

	def movePrey(self, prey, x, y):
		##self.canvas.coords(prey, self.offsetX+x*self.width/self.N, self.offsetX+y*self.height/self.N)
		self.canvas.coords(prey, self.offsetX+x*self.width/self.N, self.offsetX+y*self.height/self.N, self.offsetX+(x+1)*self.width/self.N, self.offsetY+(y+1)*self.height/self.N)

Board(10, 10, 500, 500, 11)