import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import tensorflow as tf

#%% This section reads the measured temperature in different depths to setup the boundary conditions and
# to solve for the advective flow velocity

def read_data(filename):
    df = pd.read_csv(filename)
    obs_data = df.dropna()
    obs_data=obs_data.drop(index=range(4))
    return obs_data

df=read_data(filename='DATA.csv')

df['time'] = pd.to_datetime(df['time'], format="%d.%m.%Y %H:%M:%S")
df = df.drop('depth1', axis=1)
first_time = df['time'].min()
df['time'] = df['time'] - first_time

# convert from days into seconds
df['time'] = df['time'].dt.total_seconds()/(60*60*24)
tmaxx=np.float32(df['time'].max())
tmax = dde.Variable(tmaxx)  

time = df['time'].values.reshape(-1,1)

df.set_index('time', inplace=True)

# observation depths
observe_x = np.array([0,0.21,0.42,0.62,0.82,1.01,1.22]).reshape(-1,1)
observe_T = df[['depth2','depth3','depth4','depth5','depth6','depth7','depth8']].values

LL=np.float32(observe_x[len(observe_x)-1])
L = dde.Variable(LL)


df.plot()
ax=df.plot()
ax.set_xlabel("Time [days]")
ax.set_ylabel("Temperature [°C]")
#%%
# In this section the part of the objective function is beeing defined that is associated withe the pdf (heat transport equation)

def pde(x, y):
    
    D=0.0864 #dispersion parameter in m^2/d
    a=2.091 #dimensionless factor 
    
    #Output of the NN is the temperature field T and q fluxes in time
    T, q = y[:, 0:1], y[:, 1:2] 
    
    #derivatives that are needed for the heat transport equation
    
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, component=0, i=0, j=0) 
    dy_dx = dde.grad.jacobian(y, x, i=0, j=0)
    
    return (
        dy_t
        - D * dy_xx
        + a * q * dy_dx
    )



#%%
#Define where observations are located and define IC and BC

xx, tt = np.meshgrid(observe_x,time)
X = np.vstack((np.ravel(xx), np.ravel(tt))).T #x,t
observe_T=np.ravel(observe_T).reshape(-1,1)
observe_u = dde.icbc.PointSetBC(X,observe_T,component=0)


#%% Define the model gometry in space and time

max_time=tmaxx
geom = dde.geometry.Interval(0,LL)
timedomain = dde.geometry.TimeDomain(0, max_time)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

#%
data = dde.data.TimePDE(
    geomtime,
    pde,
    [observe_u],
    num_domain=1000,
    anchors=X,
)

# Here different neural network structures are beeing tested PFNN are 
# DeepONET networks /(https://www.nature.com/articles/s42256-021-00302-5)

net = dde.nn.PFNN([2, [40, 30], [40, 30], [40, 30], [40, 30], 2], "tanh", "Glorot uniform")
#net = dde.nn.PFNN([2, [40, 30], [40, 30], [40, 30], 2], "tanh", "Glorot uniform")
#net = dde.nn.PFNN([2, [100, 100], [100, 100], [100, 100], 2], "tanh", "Glorot uniform")
#net = dde.nn.PFNN([2, [10, 10], [10, 10], [10, 10], 2], "tanh", "Glorot uniform")
#net = dde.nn.FNN([2, [200, 200], [200, 200], 2], "tanh", "Glorot uniform")
#net = dde.nn.PFNN([2, [200, 200], [200, 200], 2], "tanh", "Glorot uniform")
#net = dde.nn.PFNN([2, [50, 50], [50, 50], 2], "tanh", "Glorot uniform")


model = dde.Model(data, net)

model.compile("adam", lr=0.003,loss_weights=[1, 100])
losshistory, train_state = model.train(iterations=150000)

# I am using a two step learning procedure. Here is the second step using L-BFGS-B
model.compile("L-BFGS-B",loss_weights=[1, 1])
losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state, issave=True, isplot=True)


Y = model.predict(X)
#%%
#Plotting the different predictions vs observations

for i in range(len(observe_x)): 
    dp = ['depth2','depth3','depth4','depth5','depth6','depth7','depth8']
    columns = ['X[m]', 'Time[days]', 'T[C]', 'q[m/day]']
    sim=np.concatenate((X,Y),axis=1)
    result = pd.DataFrame(sim, columns=columns)
    
    rslt_d1 = result[result['X[m]'] == float(observe_x[i])]
    rslt_d1= rslt_d1.drop('X[m]',axis=1)
    #rslt_d1['Time[days]'] = rslt_d1['Time[days]']* tmaxx
    
    rslt_d1.plot(x='Time[days]', y='T[C]', kind='scatter', title='Temp_simulated')
    df[dp[i]].plot(kind='line', title=dp[i],color='r')

    rslt_d1.plot(x='Time[days]', y='q[m/day]', kind='line', title='Temp_simulated')














