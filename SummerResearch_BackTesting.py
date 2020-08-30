#!/usr/bin/env python
# coding: utf-8

# ### Last week, we decided that Johnson's numerical model was the best. We would like to sort of back test this by running the fit and evaluating metrics on the data set, split into 4 parts.
# ### We would also like a README that describes the methods used to reach these results.

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt    # Standard python plotting library
import os   # helps us navigate OS directories
from scipy.optimize import curve_fit   # our main fitting optimizer (reduces error while finding best fit)
from scipy import optimize # see above
import math

from scipy import signal # helps smooth our data
from scipy.signal import argrelextrema # peak finder
from scipy.signal import find_peaks # peak finder

import datetime # to keep track of run time
from scipy.integrate import odeint  # Necessary for more complex solution
from sklearn.metrics import mean_squared_error   # For analysis at the end
from scipy import stats


# In[29]:


exp = input("Input the number corresponding to the experiment you wish to analyze: (6 or 7): ")

def pull_data(n):
   file_path = 'INT_Exp'+ str(n) + '.csv'
   data = pd.read_csv(os.path.expanduser("~/Desktop/new/" + file_path))
   data.columns = ['TOF time','d','DOF time','acc_x','acc_y','acc_z','magn_x','magn_y','magn_z', 'gyr_x','gyr_y','gyr_z']
   data =  data[300:]
   data.reset_index(drop=True, inplace=True) 
   return data

df = pull_data(exp)     #all of the data is now in this df!

### Sets variable to "d" or "acc_z", important for logic when selecting analysis types later
check = False
var = 0
timer = ''
while check == False:
   acc_or_d = input("Distill & analyze acc_z (m/s.2) or distance (mm) data?: Input a or d  ")
   if acc_or_d == 'd':
      var = 'd'
      timer = 'TOF time'
      shift = 40     # Sets a higher vertical mid point for d data
      check = True
   elif acc_or_d == 'a':
      var = 'acc_z'
      timer = 'DOF time'
      shift = 10
      check = True
    


# In[30]:


def conical_pend(x, amp, g, B, shift):

   t = x.tolist()
   ax = []
   ay = []
   az = []

   ### Initial Conditions ####
   x0 = 0.388
   y0 = 0
   z0= -1.223
   x = [x0]
   y = [y0]
   z = [z0]
   vx0 = 0
   vy0 = 0.1
   vz0 = 0
   vx = [vx0]
   vy = [vy0]
   vz = [vz0]

   R = np.sqrt(x[0]**2+y[0]**2+z[0]**2) #length of pendulum in cartesian coordinates
   K = 10  # assumption: some spring constant exists in the string which causes tension in the acc_z direction

   iters =0
   try:
      for i in t:

         ts = t[iters+1]-t[iters]
         r = np.sqrt((x[iters]**2)+(y[iters]**2)+(z[iters]**2))     # radial position
         vr = (1/r)*((x[iters]*vx[iters])+(y[iters]*vy[iters])+(z[iters]*vz[iters]))     # Velocity vector dotted with r-hat
         T = g*(z[iters]/r)-(1/r)*((vx[iters]**2)+(vy[iters]**2)+(vz[iters]**2)-(vr**2))-K*(r-R)   # Tension in cartesian


         ax.append(T*(x[iters]/r) - B*vx[iters])
         ay.append(T*(y[iters]/r) - B*vy[iters])
         az.append(T*(z[iters]/r) - B*vz[iters] - g)

         x.append(x[iters] + vx[iters]*ts + (1/2)*ax[iters]*(ts**2))
         vx.append(vx[iters] + ax[iters]*ts)

         y.append(y[iters] + vy[iters]*ts + (1/2)*ay[iters]*(ts**2))
         vy.append(vy[iters] + ay[iters]*ts)

         z.append(z[iters] + vz[iters]*ts + (1/2)*az[iters]*(ts**2))
         vz.append(vz[iters] + az[iters]*ts)

         iters = iters+1
      az.reverse()
      az.append(az[-1])
      az=az+shift
      az=[(i*amp)+shift for i in az]
      return az
   except IndexError as err:
      az.reverse()
      az.append(az[-1])
      az=[(i*amp)+shift for i in az]
      return az
   except ValueError as err:
      az.reverse()
      az.append(az[-1])
      az=[(i*amp)+shift for i in az]
      return az


# # Applying this solution to the last 1/3rd of data set results in:
# - MSE = 0.0004355 , ChiSq = 0.6069
# - g = 9.67602 +/- 0.00074
# - B = 0.02894 +/- 7e-05

# # Now evenly split df into 4 parts, fit, and plot each segment.

# In[40]:


partition = round(len(df)/4)
df1 = df[:partition]
df2 = df[partition:partition+partition]
df3= df[partition+partition:partition+partition+partition]
df4=df[partition+partition+partition:]


# In[42]:


target = df1[var]
params, params_covariance = optimize.curve_fit(conical_pend, df1[timer], df1[var], p0=[0.1, 9.81, 0.02, shift], maxfev=1000)
prediction = conical_pend(df1[timer], params[0], params[1], params[2], params[3])
y = mean_squared_error(target, prediction)
chi = stats.chisquare(f_obs = target, f_exp = prediction)
print("\n")
label = "Numerical Fit for DF1: MSE=" + str(round(y,7)) + ", ChiSq=" + str(round(chi[0],4))
print("For numerical fit: g =", params[1], "+/-", round(np.sqrt(np.diag(params_covariance))[1],5))
print("B =", params[2], "+/-", round(np.sqrt(np.diag(params_covariance))[2],5))
print(label)
print("\n")


# In[47]:


target = df2[var]
params, params_covariance = optimize.curve_fit(conical_pend, df2[timer], df2[var], p0=[0.1, 9.81, 0.02, shift], maxfev=1000)
prediction = conical_pend(df2[timer], params[0], params[1], params[2], params[3])
y = mean_squared_error(target, prediction)
chi = stats.chisquare(f_obs = target, f_exp = prediction)
print("\n")
label = "Numerical Fit for DF2: MSE=" + str(round(y,7)) + ", ChiSq=" + str(round(chi[0],4))
print("For numerical fit: g =", params[1], "+/-", round(np.sqrt(np.diag(params_covariance))[1],5))
print("B =", params[2], "+/-", round(np.sqrt(np.diag(params_covariance))[2],5))
print(label)

print("\n")


# In[48]:


target = df3[var]
params, params_covariance = optimize.curve_fit(conical_pend, df3[timer], df3[var], p0=[0.1, 9.81, 0.02, shift], maxfev=1000)
prediction = conical_pend(df3[timer], params[0], params[1], params[2], params[3])
y = mean_squared_error(target, prediction)
chi = stats.chisquare(f_obs = target, f_exp = prediction)
print("\n")
label = "Numerical Fit for DF3: MSE=" + str(round(y,7)) + ", ChiSq=" + str(round(chi[0],4))
print("For numerical fit: g =", params[1], "+/-", round(np.sqrt(np.diag(params_covariance))[1],5))
print("B =", params[2], "+/-", round(np.sqrt(np.diag(params_covariance))[2],5))
print(label)

print("\n")


# In[49]:


target = df4[var]
params, params_covariance = optimize.curve_fit(conical_pend, df4[timer], df4[var], p0=[0.1, 9.81, 0.02, shift], maxfev=1000)
prediction = conical_pend(df4[timer], params[0], params[1], params[2], params[3])
y = mean_squared_error(target, prediction)
chi = stats.chisquare(f_obs = target, f_exp = prediction)
print("\n")
label = "Numerical Fit for DF4: MSE=" + str(round(y,7)) + ", ChiSq=" + str(round(chi[0],4))
print("For numerical fit: g =", params[1], "+/-", round(np.sqrt(np.diag(params_covariance))[1],5))
print("B =", params[2], "+/-", round(np.sqrt(np.diag(params_covariance))[2],5))
print(label)

print("\n")


# In[50]:





# In[ ]:




