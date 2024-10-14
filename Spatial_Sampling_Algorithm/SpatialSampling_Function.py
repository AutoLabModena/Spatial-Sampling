#--------------------------------------------------------------------------
# Spatial Sampling algorithm
#
# Written by Giovanni Braglia and Davide Tebaldi, 2024
# University of Modena and Reggio Emilia
# 
# tested on Python 3.10.12
#--------------------------------------------------------------------------

import numpy as np
import math

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
    
          tn = b*( t[j+1]-t[j] )/cn + t[j] 
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
