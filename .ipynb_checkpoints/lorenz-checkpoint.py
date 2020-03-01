import numpy as np 


class Model(object):
	"""docstring for ClassName"""
	def __init__(self):
		self.a = 0.
		self.b = 0.
		self.c = 0.
		self.tau = 0.
		self.nsteps = 0.

def rk_lorenz_solve(init_cond,ModelParams):
	t=np.zeros((ModelParams.nstep+1,))
	t[0]= 0
	y = np.zeros((3,ModelParams.nstep+1))
	y[:,0] = init_cond

	for i in range(ModelParams.nstep):
	    t[i+1]=t[i]+ModelParams.tau
	    y[:,i+1] = rk4(f, y[:,i], t[i], ModelParams)

	return(y)



def f(t,x,ModelParams):
	df = np.zeros((3,));
	df[0] = -ModelParams.a*x[0] +ModelParams.a*x[1]
	df[1] = ModelParams.b*x[0] - x[1] - x[0]*x[2]
	df[2] = -ModelParams.c*x[2] + x[0]*x[1];
	return(df)



def rk4(f, y, t, ModelParams):

	k1 = f(t,        y          , ModelParams)
	k2 = f(t + ModelParams.tau/2, y + k1*ModelParams.tau/2, ModelParams)
	k3 = f(t + ModelParams.tau/2, y + k2*ModelParams.tau/2, ModelParams)
	k4 = f(t + ModelParams.tau,   y + k3*ModelParams.tau,   ModelParams)
	y = y + ModelParams.tau/6 * (k1 + 2*k2 + 2*k3 + k4);

	return(y)