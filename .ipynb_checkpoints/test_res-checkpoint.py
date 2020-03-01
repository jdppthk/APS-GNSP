import numpy as np
from reservoir import generate_reservoir
from reservoir import reservoir_parameters
from reservoir import train_reservoir
from lorenz import rk_lorenz_solve
import matplotlib.pyplot as plt
from lorenz import Model
from glasso import keras_regressor
import scipy
from mpl_toolkits import mplot3d

ModelParams = Model() 
ModelParams.a = 10
ModelParams.b = 28
ModelParams.c = 8.0/3.0;
ModelParams.tau = 0.05
ModelParams.nstep = 20000

np.random.seed(34)

init_cond  = np.random.normal(0, 1, (3,))
generateddata = rk_lorenz_solve(init_cond, ModelParams)[:,1000:]
#generateddata = generateddata -  np.mean(generateddata, axis = 1, keepdims = True)

v = np.sqrt(np.mean(generateddata**2, axis = 1));

#generateddata[0,:] = generateddata[0,:]/v[1]
#generateddata[1,:] = generateddata[1,:]/v[1]
#generateddata[2,:] = generateddata[2,:]/v[2]

noise_scale = 0.01
generateddata = generateddata + np.random.normal(0, noise_scale, generateddata.shape)



data = np.zeros((3, generateddata.shape[1]))
data = generateddata

num_inputs = data.shape[0]
resparams = reservoir_parameters()
resparams.radius = 0.6
#plt.plot(data[0, resparams.train_length:resparams.train_length+resparams.predict_length-1])
#plt.show()
resparams.degree = 3
approx_res_size = 3000
resparams.N = int(np.floor(approx_res_size/num_inputs)*num_inputs)
resparams.sigma = 0.1
resparams.train_length = 10000
resparams.num_inputs = num_inputs;
resparams.predict_length = 5000
resparams.beta = 0.0001


[states, A, win] = train_reservoir(resparams, data)
print(states.shape)
print(data.shape)


#[history, w_out] = keras_regressor(states.T, data[0, :resparams.train_length].T, num_epochs = 2000, bsize = resparams.train_length, learn_rate = 1e-3, N = resparams.N, lam = resparams.beta)

#plt.plot(history.history['loss'])
#plt.show()

#trainout = np.dot(w_out.T, states)
#plt.plot(trainout.T)
#plt.show()

r = states[:, resparams.train_length-1]

predictions = np.zeros((3, resparams.predict_length))

#for t in range(resparams.predict_length -1):
#	out = np.dot(w_out.T, r)
#	r = np.tanh(A.dot(r) + np.dot(win,out))
#	predictions[:, t] = out

#plt.plot(predictions.T)
#plt.plot(data[0, resparams.train_length:resparams.train_length+resparams.predict_length-1])
#plt.show()


#w_out_pinv = np.linalg.pinv(states.T)*data

w_out_pinv = np.dot(np.dot(data[:, :resparams.train_length], states.T), scipy.linalg.pinv2(np.dot(states, states.T) + resparams.beta*np.eye(resparams.N))).reshape(3,-1)

for t in range(resparams.predict_length -1):
	out = np.dot(w_out_pinv, r)
	r = np.tanh(A.dot(r) + np.dot(win,out))
	predictions[:, t] = out

plt.plot(predictions.T[1:,2])
plt.plot(data[2, resparams.train_length:resparams.train_length+resparams.predict_length-1])
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')

# Data for a three-dimensional line
xline = predictions.T[1:,0]
yline = predictions.T[1:,1]
zline = predictions.T[1:,2]
ax.plot3D(xline, yline, zline, 'gray')


compare = data[:, resparams.train_length:resparams.train_length+resparams.predict_length-1]

xline = compare.T[1:,0]
yline = compare.T[1:,1]
zline = compare.T[1:,2]
ax.plot3D(xline, yline, zline, 'blue')



plt.show()