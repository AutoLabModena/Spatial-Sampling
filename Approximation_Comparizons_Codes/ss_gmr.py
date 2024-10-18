#--------------------------------------------------------------------------
# Spatial Sampling (SS) alignment and Gaussian Mixture Model (GMR) barycenter
#
# Written by Giovanni Braglia, 2024
# University of Modena and Reggio Emilia
# 
# tested on Python 3.10.12
#-------------------------------------------------------------------------- 

import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
from utils import plot_results, gmr 
from utils import SpatialSampling as ss


# Filter trajectories with SS and Load Data
#---------------------------------------------
symbol_data = np.load( 'symbol_data.npy', allow_pickle=True )
delta = 0.005
gamma = delta*1e-2
ss_series = []
for i in range(6):
  pos = symbol_data[i]['pos']
  t = np.linspace(0, 1, len(pos.T) )

  out = ss(t, pos, delta, gamma)
  tn = np.array( out.tn ).reshape(-1,1)
  sn = np.array( out.sn ).reshape(-1,1)
  xn = np.array( out.xn )
  tj = np.hstack( (tn,sn,xn) )
  ss_series.append( tj )

X_train = []
for i in range(6):
  sn = ss_series[i][:,1]
  xn = ss_series[i][:,(2,3)]
  L = len( sn ) # All curves' length normalized
  s = np.linspace( 0, 1, L )
  x = np.hstack( (s.reshape(-1,1), xn ) )
  X_train.append( x )

X_train = np.vstack( X_train )


# Fit Gaussian Mixture Model
#---------------------------------------------
gmm = mixture.BayesianGaussianMixture(
    n_components=5, # Number of Gaussian components
    covariance_type='full', # Full covariance matrices
    init_params='kmeans', # Initialize with k-means
    max_iter=150, # Max number of iterations
    weight_concentration_prior=0.01, # Control flexibility of mixture components
    random_state=0
).fit(X_train)

n_samples = 150
t = np.linspace(0, 1, n_samples )
Y_pred, Y_std  = gmr(gmm, t )


# Plots 
#---------------------------------------------
fig = plt.figure(figsize=(10, 4.5))
ax1 = plt.subplot(1,2,2)
ax2 = plt.subplot(2,2,1)
ax3 = plt.subplot(2,2,3)
for i in range(6):
  ax1.plot( ss_series[i][:,2], ss_series[i][:,3], linewidth=2, zorder=2, alpha=0.5, color='k' )
  ax2.plot( ss_series[i][:,1]/ss_series[i][-1,1], ss_series[i][:,2],  linewidth=2, zorder=1, alpha=0.5, color='k' )
  ax3.plot( ss_series[i][:,1]/ss_series[i][-1,1], ss_series[i][:,3], linewidth=2, zorder=1, alpha=0.5, color='k' )

#-----
# ax1
#-----
ax1.plot( Y_pred[:,1], Y_pred[:,2], color='r', linewidth=3., zorder=3 )
ax1.grid()
ax1.set_xticks([-0.55, -0.4])
ax1.set_yticks([-0.4, -0.2])
ax1.set_xlabel(r'$x [m]$', fontsize=18)
ax1.set_ylabel(r'$y[m]$', fontsize=18, labelpad=-25)
ax1.tick_params(axis='x', labelsize=16)
ax1.tick_params(axis='y', labelsize=16)
ax1.set_xlim([-0.6,-0.35])
ax1.set_ylim([-0.45,-0.15])

#-----
# ax2
#-----
plot_results( gmm.means_, gmm.covariances_, 1, ax2 )

ax2.plot( Y_pred[:,0], Y_pred[:,1], color='r', linewidth=3., zorder=3 )
ax2.fill_between(
    Y_pred[:,0],
    Y_pred[:,1] -  Y_std[:,1]**0.45,
    Y_pred[:,1] +  Y_std[:,1]**0.45,
    color="blue", alpha=0.1, zorder=1
    )

ax2.set_xticks([0,0.5,1])
ax2.set_yticks([-0.55, -0.4])
ax2.xaxis.set_tick_params(labelsize=16)
ax2.yaxis.set_tick_params(labelsize=16)

ax2.set_xlim([0, 1])
ax2.set_ylim([-0.6,-0.35])
ax2.grid()
ax2.set_ylabel('$x[m]$', fontsize=18, rotation='horizontal' )
ax2.yaxis.set_label_coords(-0.1, 0.4 )

#-----
# ax3
#-----
plot_results( gmm.means_, gmm.covariances_, 2, ax3 )

ax3.plot( Y_pred[:,0], Y_pred[:,2], color='r', linewidth=3., zorder=3 )
ax3.fill_between(
    Y_pred[:,0],
    Y_pred[:,2] -  Y_std[:,2]**0.45,
    Y_pred[:,2] +  Y_std[:,2]**0.45,
    color="blue", alpha=0.1, zorder=1
    )

ax3.set_xticks([0,0.5,1])
ax3.set_yticks([-0.4, -0.2])
ax3.xaxis.set_tick_params(labelsize=16)
ax3.yaxis.set_tick_params(labelsize=16)

ax3.set_xlim([0, 1])
ax3.set_ylim([-0.45,-0.15])
ax3.grid()


ax3.set_xlabel('$s[m]$', fontsize=18)
ax3.set_ylabel('$y[m]$', fontsize=18, rotation='horizontal' )
ax3.yaxis.set_label_coords(-0.1, 0.4 )

fig.suptitle('SS/GMR', fontsize=20)
fig.tight_layout()

plt.show()
