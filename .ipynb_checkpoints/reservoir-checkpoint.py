from __future__ import division
import numpy as np
import math
import scipy.sparse as sparse
import scipy.sparse.linalg as lin


class reservoir_parameters(object):
	"""docstring for ClassName"""
	def __init__(self):
		self.radius = 0.
		self.degree = 0
		self.N = 0
		self.sigma = 0.
		self.train_length = 0
		self.predict_length = 0
		self.beta = 0
		




def generate_reservoir(size, radius, degree):
	sparsity = degree/size
	A = sparse.rand(size, size, sparsity)
	e = np.max(np.abs(lin.eigs(A, k = 1, return_eigenvectors = False)))
	A = (A/e)*radius
	return(A.tocsr())


def generate_input_layer(resparams):
	q = int(resparams.N/resparams.num_inputs)
	win = np.zeros((resparams.N, resparams.num_inputs))
	for i in range(resparams.num_inputs):
	    ip = resparams.sigma*(-1 + 2*np.random.rand(q,))
	    win[i*q:(i+1)*q,i] = ip

	return(win)


def train_reservoir(resparams, data):
#	A = generate_reservoir(resparams.N, resparams.radius, resparams.degree)
	A = generate_reservoir(resparams.N, resparams.radius, resparams.degree)
	q = int(resparams.N/resparams.num_inputs)
	win = np.zeros((resparams.N, resparams.num_inputs))
	for i in range(resparams.num_inputs):
	    ip = resparams.sigma*(-1 + 2*np.random.rand(q,))
	    win[i*q:(i+1)*q,i] = ip
	states = reservoir_layer(A, win, data, resparams);
#	data = data[:,1:resparams.train_length]
	return(states, A, win)


def reservoir_layer(A, win, input, resparams):
	states = np.zeros((resparams.N, resparams.train_length))
	for i in range(resparams.train_length - 1):
		states[:,i+1] = np.tanh(A.dot(states[:,i]) + np.dot(win,input[:,i]))
	return(states)