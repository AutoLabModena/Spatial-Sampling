#--------------------------------------------------------------------------
# Spatial Sampling algorithm
#
# Written by Giovanni Braglia, 2024
# University of Modena and Reggio Emilia
# 
# tested on Python 3.10.12
#--------------------------------------------------------------------------

import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
from math import factorial
from scipy import linalg

#-------------------------------------------------------------------------------
def SpatialSampling(t, x, delta, gamma):
#-------------------------------------------------------------------------------
    """
    Spatial Sampling algorithm: given a timeseries x(t)
    return its respective parametrized curve x(s), where
    's' is the arc-length parameter.

    Parameters:
    - t: time array
    - x: positions array
    - delta: value determining the SS interval
    - gamma: tolerance w.r.t. delta

    Returns:
    - out (struct) 
       |- tn: ss-filtered time instants 
       |- xn: ss-filtered positions
       |- sn: phase variable associated with xn
       |- norm_xn: norm of xn values (=delta)
    """
    
    out = lambda: None
    out.delta = delta 

    # Transpose the input trajectory, if necessary
    if x.shape[1] > x.shape[0]:
        x = x.T

    # Initialization
    out.xn = [x[0,:]]  # Filtered trajectory
    out.sn = [t[0]]  # Arc-length trajectory
    out.tn = [t[0]]  # Time vector
    g = gamma

    xcurr = x[0, :]
    xprec = x[0, :]
    tcurr = t[0]
    j=0
  
    while j < x.shape[0]-1: 
        
        if np.linalg.norm(x[j+1,:] - xcurr) > delta+g: #INSERTION 
        #-----------------------------------------------------------
          a = xcurr - x[j,:]
          an = np.linalg.norm(a)
          
          c = x[j + 1, :] - x[j, :]  
          cn = np.linalg.norm(c)

          if math.isclose(an, 0, abs_tol=1*1e-3):
            b = delta
          else:
          # Carnot theorem
            c_alpha = a @ c / ( an*cn )

            b1 = an*c_alpha + np.sqrt( (an*c_alpha)**2 - an**2 + delta**2 )
            b2 = an*c_alpha - np.sqrt( (an*c_alpha)**2 - an**2 + delta**2 )
            if b1>0:
              b=b1
            elif b2>0:
              b=b2 
    
          tn = b*( t[j+1]-t[j] )/cn + t[j] #tcurr 
          xn = x[j,:] + b*c/cn 
          sn = out.sn[-1] + delta          

          out.tn.append(tn)
          out.xn.append(xn) 
          out.sn.append(sn) 
          
          xcurr = out.xn[-1] 
    
          continue


        if np.linalg.norm(x[j+1,:] - xcurr) <= delta+g and \
           np.linalg.norm(x[j+1,:] - xcurr) >= delta-g : #MATCH
        #-----------------------------------------------------------
          tn = t[j+1]
          xn = x[j+1,:]
          sn = out.sn[-1] + delta

          out.tn.append(tn)
          out.xn.append(xn) 
          out.sn.append(sn)

          xcurr = out.xn[-1] 

          j=j+1
          continue

        if np.linalg.norm(x[j + 1, :] - xcurr) < delta-g: #DELETION
        #-----------------------------------------------------------
          j=j+1
          continue

    # Uncomment (*) if the filtered trajectory must match final position
    # out.tn.append(t[-1]) # (*)
    # out.xn.append(x[-1, :]) # (*)
    out.norm_xn = np.zeros(len(out.xn)-1)

    for j in range(len(out.xn) - 1):
        out.norm_xn[j] = np.linalg.norm(out.xn[j + 1] - out.xn[j])

    # out.sn.append(out.sn[-1] + out.norm_xn[-1]) # (*)

    return out  


def pdf_norm_dist(val, mean_, cov_):
#-------------------------------------------------------------------------------
  """
  Calculate Normal Probability Distribution Function (pdf) value.

  Parameters:
  - val: value at which pdf should be evaluated.
  - mean_: mean value of normal pdf.
  - cov_: covariance value of normal pdf.
  
  Returns:
  - pdf: pdf value evaluated at 'val'
  """

  d = 2
  # einv = np.linalg.inv( cov_ )
  einv = 1/cov_

  a = 1/( np.sqrt( (2*np.pi)**d * np.abs(cov_) ) )
  b = np.exp( -0.5* ( val-mean_ ).T*einv*( val-mean_ ) )

  pdf = a*b

  return pdf


#-------------------------------------------------------------------------------
def gmr(gmm, X_test):
#-------------------------------------------------------------------------------
    """
    Perform Gaussian Mixture Regression (GMR).

    Parameters:
    - gmm: Fitted Gaussian Mixture Model.
    - X_test: Test input data for which to predict the output.
    - input_index: Indices of the input variables in the data.
    - output_index: Indices of the output variables in the data.

    Returns:
    - Y_pred: Predicted output data.
    """
    # Extract parameters
    means = gmm.means_
    covariances = gmm.covariances_
    pi = gmm.weights_
    reg_param = 1e-8

    n_test = X_test.shape[0]
    dim = gmm.means_.shape[1]

    Y_pred = np.zeros((n_test, dim))
    Y_std = np.zeros((n_test, dim))
    Y_pred[:,0] = X_test
    Y_std[:,0] = X_test

    for j in range(dim-1):
      for i in range(n_test):
        x_test = X_test[i]
        # Initialize the mean and covariance of the predicted distribution
        H = np.zeros(gmm.n_components)
        MuTmp = np.zeros(gmm.n_components)
        
        for k in range(gmm.n_components):
          H[k] = pi[k]*pdf_norm_dist( x_test, means[k][0], covariances[k][0,0] ) 
        H = H/np.sum(H)

        for k in range(gmm.n_components):
          MuTmp[k] = means[k][j+1] + covariances[k][j+1,0] / covariances[k][0,0] * (x_test - means[k][0])
          Y_pred[i,j+1] += H[k]*MuTmp[k]

        for k in range(gmm.n_components):
          SigmaTmp = covariances[k][j+1,j+1] - covariances[k][j+1,0] / covariances[k][0,0] * covariances[k][0,j+1]
          Y_std[i,j+1] += H[k]*( SigmaTmp + MuTmp[k]**2 )
        Y_std[i,j+1] += -Y_pred[i,j+1]**2 + reg_param

    return Y_pred, Y_std




def binomial(n, i):
#-------------------------------------------------------------------------------
  """
  Calculate binomial coefficient.

  Parameters:
  - n: total number of trials.
  - i: number of successfull trials over the total number 'n'

  Returns:
  - b: value of the binomial coefficient.
  """
  
  if (n>=0 and i>=0):
    b = factorial(n) / ( factorial(i) * factorial(n-i))
  else:
    b = 0
    
  return b



#-------------------------------------------------------------------------------
def Bernstein_pDeriv( s, p, N ):
#-------------------------------------------------------------------------------
  """
  Calculate the p-th derivative value of Bernstein Polynomial.

  Parameters:
  - s: phase variable parameter.
  - p: derivative's order (=0 for no derivative).
  - N: number of basis functions.

  Returns:
  - phi_p: value of the p-th derivative of the Bernestein Polynomial
           evaluated at s.
  """

  n = N
  g = factorial(n) / factorial(n-p)
  Tf = s[-1]
  phi_p = np.zeros( (len(s),N) )

  
  for i in range(n):
    phi = np.zeros( len(s) );
    for k in np.arange( np.max([0,i+p-n]), np.min([i,p])+1 ):
        phi +=  g*(-1)**(k+p) * binomial(p,k) * ( binomial(n-1-p,i-k) * (Tf-s)**(n-1-p-i+k) * s**(i-k) )
  
    phi_p[:,i] = phi 
    
  sum_phi = np.sum( phi_p, axis=1 ).reshape(-1,1)
  phi_p /= sum_phi

  return phi_p



#-------------------------------------------------------------------------------
def plot_results(means, covariances, ix, ax):
#-------------------------------------------------------------------------------

  splot = ax
  color_iter = itertools.cycle(["blue"]) #cornflowerblue

  for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
      covar = [ [covar[0,0], covar[0,ix]], [covar[ix,0], covar[ix,ix]] ]
      v, w = linalg.eigh(covar)
      v = 2.0 * np.sqrt(3.21887*v) # 80% confidence
      u = w[0] / linalg.norm(w[0])

      splot.scatter(mean[0], mean[ix], 25, color=color,  marker="x",  alpha=0.7, zorder=3 )

      # Plot an ellipse to show the Gaussian component
      angle = np.arctan(u[1] / u[0])
      angle = 180.0 * angle / np.pi  # convert to degrees
      ell = mpl.patches.Ellipse((mean[0], mean[ix]), v[0], v[1], angle=180.0 + angle, edgecolor=color, ls='-.',  linewidth=2, facecolor='none', zorder=3 )
      ell.set_clip_box(splot.bbox)
      ell.set_alpha(0.7)
      splot.add_artist(ell)
  return