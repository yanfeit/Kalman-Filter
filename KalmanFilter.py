"""
Kalman Filter

Copyright (C) <2019> <Yanfei Tang (yanfeit89@163.com)>

Open source software license: MIT

Permission is hereby granted, free of charge, 
to any person obtaining a copy of this software 
and associated documentation files (the "Software"), 
to deal in the Software without restriction, 
including without limitation the rights to 
use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, 
and to permit persons to whom the Software is furnished to do so, 
subject to the following conditions:
　　
The above copyright notice and this permission notice 
shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO 
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE 
AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS 
BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE 
OR OTHER DEALINGS IN THE SOFTWARE.

Credits:
The Kalman Filter code was inspired by Richard Teammco's code,
(https://www.cs.utexas.edu/~teammco/misc/kalman_filter/)

some introduction on the Kalman Filter:
http://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf
http://web.mit.edu/kirtley/kirtley/binlustuff/literature/control/Kalman%20filter.pdf
"""

from __future__ import print_function, division
import numpy as np
from numpy.linalg import inv

class KalmanFilter(object):
	"""
	Kalman Filter,
	state x is governed by the linear stochastic differential equation,
	x_(k) = A * x_(k-1) + B * u_(k-1) + w_(k-1),

	where x_(k) is the current state, for example, a point object's physical
	state is comprised by its position and velocity. 

	A is the state transition matrix

	x_(k-1) is the previous state.

	B is the control matrix related with optional control input u_(k-1)

	w_(k-1) is white noise with mean value 0 and covariance matrix Q.

	The measurement(observation) on the state x_(k) is given by
	z_(k) = H * x_(k) + v_(k)

	where z_(k) is the measurement(observation) and H is the observation matrix to
	map the real state to observation state
	
	A: 2D np_array, transition matrix
	B: 2D np_array, control matrix
	H: 2D np_array, observation matrix
	Q: 2D np_array, covariance matrix of the noise from the model
	R: 2D np_array, covaraince matrix of the noise from the measurement

	The above matrix is considered as constant here. 

	last_x: 1D np_array, previous state
	last_P: 2D np_array, previous error covariance matrix
	control: 1D np_array, optional control unit
	measureState: 1D np_array, measurement of the real state.
	x: 1D np_array, prior estimate of the state
	P: 2D np_array, prior error covariance matrix
	cur_x: 1D np_array, current estimate of the state
	cur_P: 2D np_array, current estimate error covaiance matrix
	"""

	def __init__(self, A, B, H, Q, R, cur_x, cur_P):
		
		self.A = A.copy()
		self.B = B.copy()
		self.H = H.copy()
		self.Q = Q.copy()
		self.R = R.copy()

		self.last_x = None
		self.last_P = None

		self.x = None
		self.P = None
		self.cur_x = cur_x.copy()
		self.cur_P = cur_P.copy()


	def update(self, measureState, control):

		self.measureState = measureState
		self.control = control

		# memeorize current state as previous state
		self.last_x, self.last_P = self.cur_x.copy(), self.cur_P.copy()

		# prior estimate of x, P
		self.x = np.dot(self.A, self.cur_x) + np.dot(self.B, self.control)
		self.P = np.matmul(self.A, np.matmul(self.cur_P, self.A.T)) + self.Q

		# correction
		self.S = np.matmul(self.H, np.matmul(self.P, self.H.T)) + self.R
		self.K = np.matmul(self.P, np.matmul(self.H.T, inv(self.S))) 
		self.y = self.measureState - np.dot(self.H, self.x)

		self.cur_x = self.x + np.dot(self.K, self.y)
		self.cur_P = np.matmul(np.eye(4) - np.matmul(self.K, self.H),
			self.P)




if __name__ == "__main__":

	# Test case
	A = np.array([
		[1, 0, 0.2, 0],
		[0, 1, 0, 0.2],
		[0, 0, 1, 0],
		[0, 0, 0, 1]
		])

		# no control unit
	B = np.array([
		[1, 0, 0, 0],
		[0, 1, 0, 0],
		[0, 0, 1, 0],
		[0, 0, 0, 1]
		])

		# measurement unit
	H = np.array([
		[1, 0, 0, 0],
		[0, 1, 0, 0],
		[0, 0, 1, 0],
		[0, 0, 0, 1]
		])

		# model noise
	Q = np.array([
		[0.01, 0, 0, 0],
		[0, 0.01, 0, 0],
		[0, 0, 0.01, 0],
		[0, 0, 0, 0.01]
		])

		# measurement noise
	R = np.array([
		[0.1, 0, 0, 0],
		[0, 0.1, 0, 0],
		[0, 0, 0.1, 0],
		[0, 0, 0, 0.1]
		])

	cur_x = np.array([100.0, 100.0, 0, 0])
	cur_P = np.array([
		[0, 0, 0, 0],
		[0, 0, 0, 0],
		[0, 0, 0, 0],
		[0, 0, 0, 0]
		])

	control = np.array([0, 0, 0, 0])

	model = KalmanFilter(A, B, H, Q, R, cur_x, cur_P)

	measureState = np.array([101.0, 101.2, 0.5, 0.8])

	model.update(measureState, control)

	print(model.A)
	print(model.B)
	print(model.H)
	print(model.Q)
	print(model.R)
	print("\n")
	print(model.cur_x)
	print(model.cur_P)
	print(model.last_x)
	print(model.last_P)






