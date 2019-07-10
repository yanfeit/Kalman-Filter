"""
Kalman Filter to track the mouse motion.

Copyright (c) 2019 by Yanfei Tang (yanfeit89@163.com).
Open source software license: MIT
"""

from __future__ import print_function, division
import time
import math
import numpy as np
from numpy.linalg import inv
from collections import deque, OrderedDict
from tkanimation import AnimationWindow, tk
from KalmanFilter import KalmanFilter


def _from_rgb(rgb):
	"""translates an rgb tuple of int to a tkinter friendly color code
	"""
	if type(rgb) is tuple:
		pass
	else:
		rgb = tuple(rgb)
	return "#%02x%02x%02x" % rgb

class Point(object):
	"""
	A point has life time duration.
	It will gradually fade out.
	For every frame, it will draw itself on the screen.
	"""	

	def __init__(self, window, x, y, color, lifetime):

		self.canvas = window.canvas
		self.window = window
		self.cwidth, self.cheight = int(self.canvas["width"]), int(self.canvas["height"])
		self.x = x
		self.y = y
		self.FADE_OUT_TIME = lifetime

		self.duration = int(self.FADE_OUT_TIME * window.frame_rate)
		self.maxDuration = self.duration
		self.color = color

	@property
	def lifetime(self):
		return self.FADE_OUT_TIME

	@lifetime.setter
	def lifetime(self, value):
		if 0.1 <= value <= 8.0:
			self.FADE_OUT_TIME = value	

	def  update(self):
		"""
		Update the duration of the point according to its lifetime.
		Tkinter can not assign the transparency of the color.
		To create the fading out effect, I increase the darkness of the
		color to create a fading out effect. 
		The drawback fo this rending way is that it compeltely cover the 
		underlining lines below it.
		"""

		# decrease the duration
		self.duration -= 1
		# Fade the color 
		ratio = self.duration/self.maxDuration
		self.color[0] = int(self.color[0] * ratio)
		self.color[1] = int(self.color[1] * ratio)
		self.color[2] = int(self.color[2] * ratio)

	def isAlive(self):

		return self.duration > 0

	def draw(self):
		
		# draw an oval as a point
		self.update()
		self.canvas.create_oval(self.x-2, self.y-2, self.x+2, self.y+2,
			outline = _from_rgb(self.color), fill = _from_rgb(self.color), width = 2)


class ConnectedPoint(Point):
	"""
	a variant of the Point --- two connected points (a line)
	"""
	def __init__(self, window, x, y, color, lifetime, lastx, lasty):
		"""
		Need to know the previous point position
		"""
		super(ConnectedPoint, self).__init__(window, x, y, color , lifetime)
		self.lastx = lastx
		self.lasty = lasty

	def draw(self):

		self.update()
		self.canvas.create_line(self.lastx, self.lasty, self.x, self.y,
			fill = _from_rgb(self.color), width = 4)



class KalmanFilterSimulatorWindow(AnimationWindow):
	"""
	The mouse motion tracking window and some GUI input widgets.
	"""
	def helperWidget(self):
		"""
		GUI designer function, 
		The Canvas is the main window, Some helper Widget is added
		on the side of the canvas. 
		"""
		
		# Intall Pause button, default button, restart button and a check button
		# on the frame to better organise the interface.
		self.frameTop = tk.Frame()
		self.frameTop.pack(anchor = tk.NW, side=tk.TOP, pady = 25)

		self.pauseButton = tk.Button(master = self.frameTop, text="Pause", width = 8, 
			command = self.pauseClick)
		self.pauseButton.pack(anchor=tk.NE, side=tk.LEFT, padx= 5)

		self.defaultButton = tk.Button(master = self.frameTop, text="Restore Default Values", 
			command = self.restoreClick)
		self.defaultButton.pack(anchor=tk.NW, side=tk.LEFT, padx= 5)

		self.restartButton = tk.Button(master = self.frameTop, text="Restart", 
			command = self.restartClick)
		self.restartButton.pack(anchor=tk.NW, side=tk.LEFT, padx= 5)

		self.showMouseTraceButtonOn = tk.BooleanVar()
		self.showMouseTraceButton = tk.Checkbutton(master = self.frameTop, text="Show Mouse Trace", 
			variable = self.showMouseTraceButtonOn, command = self.showMouseTrace)
		self.showMouseTraceButton.pack(anchor=tk.NW, side=tk.LEFT, padx= 5)

        # setup the instance of Entry
		self.entries = OrderedDict()
		self.entries["A"] = [[0] * 4 for i in range(4)]
		self.entries["B"] = [[0] * 4 for i in range(4)]
		self.entries["H"] = [[0] * 4 for i in range(4)]
		self.entries["Q"] = [[0] * 4 for i in range(4)]
		self.entries["R"] = [[0] * 4 for i in range(4)]
		self.entries["N"] = [[0] * 4 for i in range(4)]
		self.code2entries = {0:"A", 1:"B", 2:"H", 3:"Q", 4:"R", 5:"N"}

		# Install Entries of Matrix A, B, H on the medium top frame beneath the top frame.

		self.frameMediumTop = tk.Frame()
		self.frameMediumTop.pack(anchor = tk.NW, side=tk.TOP, pady = 10)

		for _ in range(3):
			# Frame1 hold entries of matrix A, B, H in each iteration
			self.frame1 = tk.Frame(master = self.frameMediumTop)
			self.frame1.pack(anchor = tk.NW, side=tk.LEFT, padx = 10)

			for i in range(5):
				self.frame2 = tk.Frame(master = self.frame1)
				self.frame2.pack(anchor = tk.NW, side=tk.TOP)
				if i == 4:

					if _ == 0:
						label = tk.Label(master = self.frame1, text = "A matrix\nState Transition")
						label.pack(anchor = tk.NW, side=tk.LEFT, padx = 2)
					elif _ == 1:
						label = tk.Label(master = self.frame1,text = "B matrix\nInput Control")
						label.pack(anchor = tk.NW, side=tk.LEFT, padx = 2)
					elif _ == 2:
						label = tk.Label(master = self.frame1,text = "H matrix\nMeasurement")
						label.pack(anchor = tk.NW, side=tk.LEFT, padx = 2)

				else:

					for j in range(4):
						self.entries[self.code2entries[_]][i][j] = tk.Entry(master = self.frame2, width = 3)
						self.entries[self.code2entries[_]][i][j].pack(
                            anchor=tk.NW, side=tk.LEFT, padx= 1)

		# Install Entries of Matrix Q, R, N on the medium top frame beneath the medium top frame.				
		self.frameMediumBottom = tk.Frame()
		self.frameMediumBottom.pack(anchor = tk.NW, side=tk.TOP)

		for _ in range(3, 6):
			# Frame1 hold entries of matrix Q, R, N in each iteration
			self.frame1 = tk.Frame(master = self.frameMediumBottom)
			self.frame1.pack(anchor = tk.NW, side=tk.LEFT, padx = 10)

			for i in range(5):
				self.frame2 = tk.Frame(master = self.frame1)
				self.frame2.pack(anchor = tk.NW, side=tk.TOP)
				if i == 4:

					if _ == 3:
						label = tk.Label(master = self.frame1, text = "Q matrix\nAction Uncertainty")
						label.pack(anchor = tk.NW, side=tk.LEFT, padx = 2)
					elif _ == 4:
						label = tk.Label(master = self.frame1,text = "R matrix\nSensor Noise")
						label.pack(anchor = tk.NW, side=tk.LEFT, padx = 2)
					elif _ == 5:
						label = tk.Label(master = self.frame1,text = "N matrix\nDistrubation")
						label.pack(anchor = tk.NW, side=tk.LEFT, padx = 2)

				else:

					for j in range(4):
						self.entries[self.code2entries[_]][i][j] = tk.Entry(master = self.frame2, width = 3)
						self.entries[self.code2entries[_]][i][j].pack(
                            anchor=tk.NW, side=tk.LEFT, padx= 1)


		# Install a slider bar on the bottom frame
		self.frameBottom = tk.Frame()
		self.frameBottom.pack(anchor = tk.NW, side=tk.TOP, pady = 50)

		label = tk.Label(master = self.frameBottom, text = "Fade-Out Time (sec):")
		label.pack(anchor = tk.NW, side=tk.TOP, padx = 10, pady = 0)

		# Life time for the fading-out points
		self.lifetime = 4
		self.lifetimeScale = tk.Scale(self.frameBottom, from_=0.1, to=8, length = 200, orient=tk.HORIZONTAL, resolution=0.1, command=self.onScale)
		# initialize the scale
		self.lifetimeScale.set(self.lifetime)
		self.lifetimeScale.pack(anchor = tk.NW, side=tk.TOP, padx = 10, pady = 0)
    

	def setup(self):
		"""
		Setup the intial condition for the Kalman Filter, 
		"""

		self.mouseX, self.mouseY = 0, 0
		self.premouseX, self.premouseY = 0, 0

		# delta_t 
		self.A = np.array([
			[1, 0, self.frame_time, 0],
			[0, 1, 0, self.frame_time],
			[0, 0, 1, 0],
			[0, 0, 0, 1]
			])

		# no control unit
		self.B = np.array([
			[1, 0, 0, 0],
			[0, 1, 0, 0],
			[0, 0, 1, 0],
			[0, 0, 0, 1]
			])

		# measurement unit
		self.H = np.array([
			[1, 0, 0, 0],
			[0, 1, 0, 0],
			[0, 0, 1, 0],
			[0, 0, 0, 1]
			])

		# model noise
		self.Q = np.array([
			[0.01, 0, 0, 0],
			[0, 0.01, 0, 0],
			[0, 0, 0.01, 0],
			[0, 0, 0, 0.01]
			])

		# measurement noise
		self.R = np.array([
			[0.1, 0, 0, 0],
			[0, 0.1, 0, 0],
			[0, 0, 0.1, 0],
			[0, 0, 0, 0.1]
			])

		# Noise to disturb the true position of the mouse
		self.N = np.array([
			[100, 0, 0, 0],
			[0, 100, 0, 0],
			[0, 0, 0, 0],
			[0, 0, 0, 0]
			])

		# current state
		self.cur_x = np.array([self.mouseX, self.mouseY, 0, 0])

		# current error covariance matrix
		self.cur_P = np.array([
			[0, 0, 0, 0],
			[0, 0, 0, 0],
			[0, 0, 0, 0],
			[0, 0, 0, 0]
			])

		# Record the default value.
		self.Adefault = self.A.copy()
		self.Bdefault = self.B.copy()
		self.Hdefault = self.H.copy()
		self.Qdefault = self.Q.copy()
		self.Rdefault = self.R.copy()
		self.Ndefault = self.N.copy()

		# control unit, For this case there is no control unit
		self.control = np.array([0, 0, 0, 0])

		# rPoint: record the measurement, the measurement is created
		#         by the true hidden state with Gaussian noise
		# kPoint: store the positions of points from estimation of the Kalman filter
		# tPoint: record true position of the mouse
		self.rPoints = deque()
		self.kPoints = deque()
		self.tPoints = deque()
		# self.pPoints = deque()


		self.running = True

		# I commented out the prediction line as it seems not very useful from my point of view.
		# But I add the true mouse position.
		# self.predictionOn = True

		self.mousepositionOn = self.showMouseTraceButtonOn.get()

		# Instance of the Kalman filter, 
		self.kfmodel = KalmanFilter(self.A, self.B, self.H, self.Q, self.R, 
			self.cur_x, self.cur_P)

		# Intialize the entries, 
		# TO THINK: there is a better way? better to put it in a helperWidget() fucntion,
		#           as this function is used to design graphic interface...
		self.setMatrix("A", self.Adefault)
		self.setMatrix("B", self.Bdefault)
		self.setMatrix("H", self.Hdefault)
		self.setMatrix("Q", self.Qdefault)
		self.setMatrix("R", self.Rdefault)
		self.setMatrix("N", self.Ndefault)


	def draw(self):
		"""
		Update the canvas in each frame per second.
		"""

		if not self.running:
			return
		
		# Clean the Canvas first.
		self.canvas.delete(tk.ALL)

		# Receive the values from the entries of each matrix.
		# Is there a better to do that? As if I don't modify the matrix,
		# I don't need to read the entries every time. 
		# Is there a callback function of the Entry object? NO..., 
		# https://stackoverflow.com/questions/6548837/how-do-i-get-an-event-callback-when-a-tkinter-entry-widget-is-modified
		# There is a solution, But it adds some complexities. I don't like it.
		self.modifyMatrix("A")
		self.modifyMatrix("B")
		self.modifyMatrix("H")
		self.modifyMatrix("Q")
		self.modifyMatrix("R")
		self.modifyMatrix("N")

		# the core of this code... ....
		#**************** Kalman Filter *******************#
		# Hidden State
		self.curState = np.array([self.mouseX, self.mouseY, 
			(self.mouseX - self.premouseX)/self.frame_time, 
			(self.mouseY - self.premouseY)/self.frame_time])
		# Gaussian Noise
		self.measureNoise = np.random.multivariate_normal(
			[0, 0, 0, 0], self.N)
		# Apply measurement, z_k = H_k * x_k + V_k
		self.measureState = np.dot(self.H, self.curState) + self.measureNoise
		
		self.kfmodel.update(self.measureState, self.control)
		#***************       END       ******************#
		

		# Store the measured, estimated, and mouse position points
		# to the deque
		self.rPoints.append( Point(self, self.measureState[0], 
			self.measureState[1], [255, 255, 255],  self.lifetime))

		self.kPoints.append( ConnectedPoint(self, 
			self.kfmodel.cur_x[0], self.kfmodel.cur_x[1], [0, 255, 0], self.lifetime,
			self.kfmodel.last_x[0], self.kfmodel.last_x[1]) )

		if self.mousepositionOn:
			self.tPoints.append( ConnectedPoint(self,
				self.mouseX, self.mouseY, [0, 0, 255], self.lifetime,
				self.premouseX, self.premouseY))

		# Draw points in the deque, pop out those points which are dead
		def drawPoints(pointDeque):
			count_dead = 0
			for point in pointDeque:
				if point.isAlive():
					point.draw()
				else:
					count_dead += 1
			for _ in range(count_dead):
				pointDeque.popleft()
			
		drawPoints(self.rPoints)
		drawPoints(self.kPoints)
		if self.mousepositionOn:	
			drawPoints(self.tPoints)
		###############################

		# Not very useful... to show the status of state.
		#self.showStatus()

		# keep track of the current mouse position as previous mouse position
		self.premouseX, self.premouseY = self.mouseX, self.mouseY



	def mousemotion(self, mouseposition):
		"""
		store the mouse position 
		"""
		self.mouseX, self.mouseY = mouseposition[0], mouseposition[1]

	def pauseClick(self):
		"""
		pause or resume animation
		"""
		self.running = not self.running
		if self.running:
			self.pauseButton.config(relief = tk.RAISED, text="Pause")
		else:
			self.pauseButton.config(relief = tk.SUNKEN, text="Resume")

	def restartClick(self):
		"""
		restart the animation
		"""
		self.setup()

	def restoreClick(self):
		"""
		restore to the default value...
		"""
		self.lifetime = 4
		self.lifetimeScale.set(self.lifetime)

		self.setMatrix("A", self.Adefault)
		self.setMatrix("B", self.Bdefault)
		self.setMatrix("H", self.Hdefault)
		self.setMatrix("Q", self.Qdefault)
		self.setMatrix("R", self.Rdefault)
		self.setMatrix("N", self.Ndefault)


	def showMouseTrace(self):
		"""
		toggle the trace of mouse
		"""
		if self.showMouseTraceButtonOn.get() == True:
			self.mousepositionOn = True
		else:
			self.mousepositionOn = False

	def onScale(self, value):
		"""
		Fade out time
		"""
		self.lifetime = float(value)

	def setMatrix(self, id, vals):
		"""
		set the values of the matrix
		"""
		for i in range(len(vals)):
			for j in range(len(vals[0])):
				self.entries[id][i][j].delete(0, tk.END)
				self.entries[id][i][j].insert(0, vals[i,j])

	def modifyMatrix(self, id):
		"""
		modify the matrix according to the GUI input.
		"""
		switcher = {"A": self.A,
					"B": self.B,
					"H": self.H,
					"Q": self.Q,
					"R": self.R,
					"N": self.N}

		for i in range(4):
			for j in range(4):
				# The following state make sure even the entry is empty, it
				# can still work. Alphabic characters are not alowed. 
				if self.entries[id][i][j].get() == "":
					switcher[id][i, j] = 1.0
				else:	
					switcher[id][i, j] =  float(self.entries[id][i][j].get())

	def showStatus(self):
		"""
		Print the status of the mouse (position and velocity), 
		the Kalman filter estimation of the status of the mouse
		"""
		statustext = """Mouse status, (x, y, velx, vely) = ({0:7.0f}, {1:7.0f}, {2:7.2f}, {3:7.2f})\n
Kalman filter estimation            = ({4:7.0f}, {5:7.0f}, {6:7.2f}, {7:7.2f})\n""".format(
							self.curState[0], self.curState[1], self.curState[2], self.curState[3],
							self.kfmodel.cur_x[0], self.kfmodel.cur_x[1], self.kfmodel.cur_x[2], self.kfmodel.cur_x[3])

		self.canvas.create_text(0, 150, text = statustext, fill = "green4", anchor = tk.NW)




if __name__ == "__main__":
	window = KalmanFilterSimulatorWindow(800, 800, "Kalman Filter")
	window.mainloop()