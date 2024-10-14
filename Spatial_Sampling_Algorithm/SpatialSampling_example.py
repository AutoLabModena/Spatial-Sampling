#--------------------------------------------------------------------------
# Spatial Sampling example code
#
# Written by Giovanni Braglia, 2024
# University of Modena and Reggio Emilia
# 
# tested on Python 3.10.12
#--------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from SpatialSampling_Function import SpatialSampling as ss

# Load data from Panda Co-Manipulation Dataset
#---------------------------------------------
symbol_data = np.load( "symbol_data.npy", allow_pickle=True )
Ts = 0.001 # Sampling-time of the recording
i = 2 # i = [0,1,...,5]
L = symbol_data[i]['pos'].shape[1]
T = Ts*L # Duration of the recording
t = np.linspace( 0, T, L )
pos = symbol_data[i]['pos'] # Position data

# Spatial Sampling
#---------------------------------------------
delta = 0.005 # Spatial Sampling's interval
gamma = delta*1e-3 # Delta tolerance
out = ss( t, pos, delta, gamma )

tn = np.array( out.tn ).reshape(-1,1)
sn = np.array( out.sn ).reshape(-1,1)
xn = np.array( out.xn )

# Plots
#---------------------------------------------
plt.figure(figsize=(6,8), constrained_layout=True )

ax0 = plt.subplot(2,1,1)
#---------------------------
plt.plot( symbol_data[i]['pos'][0,:], symbol_data[i]['pos'][1,:], 'k-', linewidth=2, label='Original' )
plt.plot( xn[:,0], xn[:,1], 'r--', linewidth=2, label='Filtered' )
ax0.legend()
ax0.grid()
ax0.set_xlabel(r'$x$[m]', fontsize=14 )
ax0.set_ylabel(r'$y$[m]', fontsize=14 )

ax1 = plt.subplot(3,1,3)
#---------------------------
plt.plot( tn, sn, 'r-', linewidth=2 )
ax1.grid()
ax1.set_xlabel(r'$t$[s]', fontsize=14 )
ax1.set_ylabel(r'$s$[m]', fontsize=14 )

plt.show()

