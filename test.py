import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from matplotlib.ticker import LinearLocator

def reward(z,vz):
    print('Exponential: ',vz * np.tanh((z-0.2)*5)+1.3)
    print('Base: ',(-1.1 * np.tanh((z-0.2)*5)))
    r=-np.tanh((z-0.2)*5)*(1.1  )**(vz * np.tanh((z-0.2)*5)+1.3)
    return r

z_array = np.linspace(0,2.5,100)
vz_array = np.linspace(-3,0,100)
Z, VZ = np.meshgrid(z_array, vz_array)
R = 5*(1.5**(.5*Z/(VZ-0.2)+5*(VZ-.11)))
print(R)
R = np.clip(R, -1, 5)
# R= -np.tanh((Z-0.2)*5)*(1.1 )**(VZ * np.tanh((Z-0.2)*5)+1.3)
print(reward(2.4,0))
# plot 3d graph
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(Z, VZ, R, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

plt.show()
