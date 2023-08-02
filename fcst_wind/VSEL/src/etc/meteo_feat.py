import os, sys
import numpy as np
import matplotlib as mpl
import tensorflow as tf
mpl.use('tkagg')
import matplotlib.pyplot as plt
## Local files
import boruta_py as boruta 
import fs_mj as fs

## Load data
nwp = np.load("../dat/test_nwp_test1stn.npz")['value']
#obs = np.load("../dat/test_obs_test1stn.npz")['value']
obs = np.load("../dat/test_nwp_test1stn.npz")['value'][:,0]
print("read obs shape: ", obs.shape)
header = np.load("../dat/test_nwp_test1stn.npz")['value_name']
header = header.tolist()

print (np.shape(nwp), np.shape(obs), header, type(header)) # variable, day, forecast_time

### Data: pre_process ##
#pnwp, pobs = fs.Remove_nan_batch(nwp[:,0,:,:], obs[:,0,:,:])
#
### Data_process ###
#varib, period, time = pnwp.shape #nwp.shape
#
#md_cov=list()
#for i in list(range(varib)): #range(nwp.shape[0])): ## variable
# md_all=list()
# for j in list(range(period)): # to 156/ nwp.shape[2]
#  for k in list(range(time)): # np.arange(0,6,1) ## forecast
#   md_all.append(pnwp[i][j][k]) ## station
# md_cov.append(md_all)
#data_np = np.array(md_cov).T
#
### OBS ###
#obs_cov=list()
#for j in list(range(period)): # to 156/ nwp.shape[2]
# for k in list(range(time)): # np.arange(0,6,1) ## forecast
#  obs_cov.append(pobs[0][j][k])
#target_np = np.array(obs_cov).T

data_np = nwp
target_np = obs

print (data_np, target_np, np.shape(data_np), np.shape(target_np)) # forecast_time * day, variable
#plt.boxplot(data_np)

###################################################################
### Boruta method
##with tf.device('/device:GPU:0'):
## threshold, scores, index = fs.BoruTa(data_np, target_np, 'continous') # input_header, input, type : 'continuous' or 'discrete'
##header.append(threshold)
##fs.importance_plot(header, scores)
##new_np, new_head = fs.selected_features(data_np, header, scores, index) # selected/
##print ('Importance : ', scores)
##print ('Ranking : ', index) # 1: selected, 2 : tentative, 3,4,5 : unselected
#
### Mutual Information
##with tf.device('/device:GPU:1'):
## scores, index = fs.MI(header, data_np, target_np, 3) # final = kvalue
##print (index, len(scores), len(header))
##fs.importance_plot(header, scores) # plot
##new_np, new_head = fs.selected_features(data_np, header, scores, index) # selected/
#
## Gradient Boosting
##with tf.device('/device:GPU:1'):
#
scores, index = fs.GradBoosting(data_np, target_np, 1) # final = rand_st
print (len(scores), len(header))
fs.importance_plot(header, scores)
new_np, new_head = fs.selected_features(data_np, header, scores, index) # selected/

####################################################################

## Correlation
print (np.shape(new_np))

## covariance maxtrix
print("before in cov_matrix2", new_np.shape)
Cov_mat, mean, sx, sy = fs.Cov_matrix2(new_np)

## correlation
Correlate = fs.Correlation(Cov_mat, sx, sy)
# Correlation = Correlation2(tv)
print("after in correlation", Correlate.shape)

# heatmap
fs.heatmap(Correlate, new_head, new_head, np.min(Correlate))  ## (value, X_axis, Y_axis)
