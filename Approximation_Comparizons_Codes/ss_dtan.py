#--------------------------------------------------------------------------
# Spatial Sampling (SS) alignment and Diffeomorphic Temporal Alignment 
# Nets (DTAN) barycenter
#
# Written by Giovanni Braglia, 2024
# University of Modena and Reggio Emilia
# 
# tested on Python 3.10.12
#-------------------------------------------------------------------------- 

import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
from utils import Bernstein_pDeriv 
from utils import SpatialSampling as ss
import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# !!!!!! CHANGE TO YOUR dtan FOLDER !!!!
sys.path.append('/YOUR/PATH/TO/dtan')
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
from DTAN.DTAN_layer import DTAN as dtan_model


# Filter trajectories with SS and Load Data
#---------------------------------------------
symbol_data = np.load( 'symbol_data.npy', allow_pickle=True )
delta = 0.005
gamma = delta*1e-2

# Bernstein polynomials parameters
n_samples = 150
Nb = 15 # Number of basis functions
s = np.linspace(0, 1, n_samples) # All curves' length normalized
Psi = Bernstein_pDeriv( s, 0, Nb )
ss_ds = np.zeros( (6,2,n_samples) )

for i in range(6):
  pos = symbol_data[i]['pos']
  t = np.linspace(0, 1, len(pos.T) )

  out = ss(t, pos, delta, gamma)
  tn = np.array( out.tn ).reshape(-1,1)
  sn = np.array( out.sn ).reshape(-1,1)
  xn = np.array( out.xn )
  
  # Resample trajectories to have same length
  n_tmp   = len( sn )
  s_tmp = np.linspace( 0, 1, n_tmp )
  Psi_tmp = Bernstein_pDeriv( s_tmp, 0, Nb )
  wx = np.linalg.pinv( Psi_tmp ) @ xn[:,0]
  wy = np.linalg.pinv( Psi_tmp ) @ xn[:,1]
  x_tmp = Psi@wx
  y_tmp = Psi@wy
  ss_ds[i] = np.hstack( ( x_tmp.reshape(-1,1), y_tmp.reshape(-1,1) ) ).T


# Train DTAN network
#---------------------------------------------
num_epochs = 50
X_train_tensor = torch.Tensor(ss_ds)
train_dataset = TensorDataset(X_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = dtan_model(signal_len=n_samples, channels=2, n_recurrence=1, zero_boundary=True, device='cpu').to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for k, inputs in enumerate(train_loader):
        inputs = inputs[0].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

aligned_trajectories = []
model.eval()
with torch.no_grad():
    for inputs in train_loader:
        inputs = inputs[0].to(device)
        aligned_output = model(inputs)
        aligned_output = aligned_output.cpu().numpy()

        for traj in aligned_output:
            aligned_trajectories.append(traj)

aligned_trajectories = np.array(aligned_trajectories)
bar = np.mean(aligned_trajectories, axis=0) # Barycenter



# Plots 
#---------------------------------------------
plt.close('all')
fig = plt.figure(figsize=(10, 4.5))
ax1 = plt.subplot(1,2,2)
ax2 = plt.subplot(2,2,1)
ax3 = plt.subplot(2,2,3)
for i in range(6):
  ax1.plot( aligned_trajectories[i][0,:], aligned_trajectories[i][1,:], linewidth=2, zorder=2, alpha=0.5, color='k' )
  ax2.plot( s, aligned_trajectories[i][0,:],  linewidth=2, zorder=1, alpha=0.5, color='k' )
  ax3.plot( s, aligned_trajectories[i][1,:], linewidth=2, zorder=1, alpha=0.5, color='k' )

#-----
# ax1
#-----
ax1.plot( bar[0,:], bar[1,:], color='r', linewidth=3., zorder=3 )
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
ax2.plot( s, bar[0,:], color='r', linewidth=3., zorder=3 )
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
ax3.plot( s, bar[1,:], color='r', linewidth=3., zorder=3 )
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

fig.suptitle('SS/DTAN', fontsize=20)
fig.tight_layout()

plt.show()
