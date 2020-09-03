import pandas as pd
import numpy as np

import matplotlib.pyplot as plt    # Standard python plotting library
import os   # helps us navigate OS directories
from scipy.optimize import curve_fit   # our main fitting optimizer (reduces error while finding best fit)
from scipy import optimize # see above
import math

import datetime # to keep track of run time
from sklearn.metrics import mean_squared_error   # Calculates error between fit and data
from scipy import stats  # Calculates ChiSq (goodness of fit)

exp = input("Input the number corresponding to the experiment you wish to analyze: (6 or 7): ")

def pull_data(n):
   file_path = 'INT_Exp'+ str(n) + '.csv'
   data = pd.read_csv(os.path.expanduser("~/Desktop/new/" + file_path))
   data.columns = ['TOF time','d','DOF time','acc_x','acc_y','acc_z','magn_x','magn_y','magn_z', 'gyr_x','gyr_y','gyr_z']
   data =  data[300:]
   return data


timer = 'DOF time'   # We only use DOF times, since it corresponds to acc_z data!
df = pull_data(exp)     #all of the data is now in this df!
df = df.loc[(df[timer] >= 200)] # Lets only analyze good data (Ignore the first 200 seconds due to huge decay)


###---Defining our numerical solution based model!---###
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

#### plotting our models with metrics ####
var = "acc_z"     # our variable of interest!
shift = 10       # The data rests at roughly a midpoint of 10 on the y-axis
target = df[var] # Actual Acc_z Values

params, params_covariance = optimize.curve_fit(conical_pend, df[timer], target,
                                               p0=[0.1, 9.81, 0.02, shift], maxfev=1000)  # We help python with an initial guess of G = 9.81 and B = 0.02


prediction = conical_pend(df[timer], params[0], params[1], params[2], params[3])
# We store our predicted acc_z or d values in the array above


# Calculating errors
y = mean_squared_error(target, prediction)
chi = stats.chisquare(f_obs = target, f_exp = prediction)

#formatting everything for results.
print("\nPredicted Values")
print("----------------------------------------------------------------------")
label = "Numerical Fit: MSE=" + str(round(y,7)) + ", ChiSq=" + str(round(chi[0],4))
print("g =", params[1], "+/-", round(np.sqrt(np.diag(params_covariance))[1],5))
print("B =", params[2], "+/-", round(np.sqrt(np.diag(params_covariance))[2],5))
print("\n")


#Finally plotting the fit over our raw data
plt.figure(figsize=(16,8))
plt.rc('axes', labelsize=14)
plt.plot(df[timer], df[var], label='Raw Data', alpha = 0.5)
plt.plot(df[timer], prediction, label=label, alpha=0.9)
plt.xlabel("time(s)")
plt.ylabel(str(var))

plt.title("Results of fit", fontsize="xx-large")
plt.legend(loc='best')
plt.show()
