import os,sys
#import pandas as pd
#import glob
#import time,copy
#import csv
#import math
#from operator import itemgetter
import boruta_py as boruta 

## Correlation (Pereson)
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid

## MI
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn import metrics

#----------------------------------------------------------#
# Nahum 1:7
# The LORD is good, a strong hold in the day of trouble; 
# and he knoweth them that trust in him. 
# M.J. Song (2021.05.17)
#----------------------------------------------------------#

###---------------------- Correlation ----------------------###
def Cov_matrix(input):
    print (input)
    mean = [ np.mean(i) for i in input ] ## Mean 
    variance = [ (np.array(i)-mean[k]) for k, i in enumerate(input) ] ## Variance

    ## Product covariance matrix
    cov = np.zeros([len(input), len(input)])
    for k1, i in enumerate(variance) :
     for k2, j in enumerate(variance) :
      covli=list()
      covli.append(i*j)  ## covariance (array * array)
      sum = 0.0
      for k in list(range(len(covli[0]))) :
       sum = sum + covli[0][k]
      CN = len(covli[0])
      cov[k1,k2] = sum/CN

    ## Standard deviation
    Sx = np.zeros([len(input), len(input)]); Sy = np.zeros([len(input), len(input)])
    for k1, i in enumerate(variance):
     for k2, j in enumerate(variance):
      correl_1=list(); correl_2=list()
      correl_1.append(i*i); correl_2.append(j*j)
      sum_1 = 0.0; sum_2 = 0.0
      for k in list(range(len(correl_1[0]))):
       sum_1 = sum_1 + correl_1[0][k]
       sum_2 = sum_2 + correl_2[0][k]
      CoN = len(correl_1[0])
      Sx[k1,k2] = sum_1/CoN
      Sy[k1,k2] = sum_2/CoN
    return cov, mean, Sx, Sy

def Cov_matrix2(input):
     #input = np.array(input) ## Input
    print (input)
    mean = [ np.mean(i) for i in input ] ## Mean 
    variance = [ (np.array(i)-mean[k]) for k, i in enumerate(input) ] ## Variance

    ## Product covariance matrix
    cov = np.zeros([len(input), len(input)])
    for k1, i in enumerate(variance):
     for k2, j in enumerate(variance):
      cov[k1,k2] = np.sum((i*j), dtype=np.float128)/len(input[0])
    print ('Covariance_matrix', cov)

    ## Standard deviation
    Sx = np.zeros([len(input), len(input)]); Sy = np.zeros([len(input), len(input)])
    for k1, i in enumerate(variance):
     for k2, j in enumerate(variance):
      Sx[k1,k2] = np.sum((i**2), dtype=np.float128)/len(input[0])
      Sy[k1,k2] = np.sum((j**2), dtype=np.float128)/len(input[0])
     #print ('Sx', cov[1,1], Sx[1,1], Sy[1,1])
    return cov, mean, Sx, Sy

def Correlation(cov, sx, sy):
    print (cov)
    cor = np.zeros(cov.shape)
    for i in range(cov.shape[0]):
     for j in range(cov.shape[1]):
        if (cov[i][i] == 0 or cov[j][j]  == 0):
         cor[i][j] = 0
        else:
         cor[i][j] = round(cov[i][j]/(np.sqrt(sx[i][i])*np.sqrt(sy[j][j])),2) ## Correlation
    print ('Correlation', cor)
    return cor

def Correlation2(input):
    R1 = np.corrcoef(input)
    print (R1)
    return R1

###------------------------------ Feature selection --------------------------###
# Regressor
def MI(header, data_np, target_np, k_cnt): # Mutual information
    print ('##Mutual Information ##')
    feat_start=1
    sel=SelectKBest(mutual_info_regression, k=k_cnt)
    fit_mod=sel.fit(data_np, target_np)
    print ('Univariate Feature Selection - Mutual Info: ', ' K : ', k_cnt)
    ##sel_idx=fit_mod.get_support()

    #Print ranked variables out sorted
    temp=[]
    scores=fit_mod.scores_
    #print ('scores :', scores, feat_start, header)

    sel_idx=list()
    for i in list(range(len(scores))) :
     if (scores[i] > np.mean(scores)) :
      sel_idx.append(True)
     else:
      sel_idx.append(False)
    print ('scores by MI :', scores)
    print('\n')
    return scores, sel_idx

def GradBoosting(data_np, target_np, rand_st): # Gradient Boosting
    #Wrapper Select via model
    print ('## Gradient Boosting ##')
    #rgr = GradientBoostingRegressor(loss='ls', learning_rate=0.01, n_estimators=100,
    rgr = GradientBoostingRegressor(loss='ls', learning_rate=0.01, n_estimators=2,
                                min_samples_split=3, max_depth=3, random_state=rand_st)
    sel = SelectFromModel(rgr, prefit=False, threshold='mean', max_features=None)
    rgr.fit(data_np, target_np)
    print ('Gradient_boosting (Regressor): ')

    fit_mod=sel.fit(data_np, target_np)
    sel_idx=fit_mod.get_support()
    scores = rgr.feature_importances_
    print ('scores by gradboosting : ', scores)
    print('\n')
    return scores, sel_idx

def BoruTa(data_np, target_np, type): # boruta
    # define random forest classifier, with utilising all cores and
    # sampling in proportion to y labels
    print ('## Boruta regressor(Randomforest) ##')
    if (type == 'continous'):
     rf = RandomForestRegressor(n_jobs=-1, criterion='mse', max_depth=30,  ## max_depth=5
                                min_samples_split=3, max_features=0.33, random_state=1)
    else:
     rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

    # define Boruta feature selection method
    # perc: Instead of the max we use the percentile defined by the user, to pick our threshold for comparison between shadow and real features. The max tend to be too stringent. This provides a finer control over this. The lower perc is the more false positives will be picked as relevant but also the less relevant features will be left out. The usual trade-off. The default is essentially the vanilla Boruta corresponding to the max.
    print ('## Boruta_py ##')
    feat_selector = boruta.BorutaPy(rf, n_estimators='auto', perc=100, verbose=2, random_state=1)

    # find all relevant features - 5 features should be selected
    feat_selector.fit(data_np, target_np)

    # check selected features - first 5 features are selected
    feat_selector.support_
     #print (feat_selector.support_)

    # Scores
    threshold = np.median(feat_selector.shadow_history)
    importance = feat_selector.importance[-1]

    # Ranking
    ranking = feat_selector.ranking_
    ranks=list()
    for i in ranking:
     if (i > 1):
      ranks.append(False)
     else:
      ranks.append(True)

    return threshold, importance, ranks

def selected_features(data_np, header, scores, index):
    head=[]; temp_idx=[]; temp_del=[]
    for i in list(range(len(header))):
     if index[i]==1:             #Selected Features get added to temp header
      head.append(header[i])
      temp_idx.append(i)
     else:                      #Indexes of non-selected features get added to delete array
      temp_del.append(i)
    print('Selected', head)
    print('Features (total/selected):', len(index), len(head))
    print('\n')

    ## Delete variable not contained to our purpose
    for field in head:
     header.append(field)
    data_np = np.delete(data_np, temp_del, axis=1)
    print (head, data_np, np.shape(data_np))

    data = np.array(data_np).T
    return data, head

###---------------------------- Normalization --------------------------------###
def min_max_1d(X):
    Num=np.shape(X)[0]
    maxlist = np.amax(X,axis=0); minlist = np.amin(X,axis=0)
        
    B = maxlist-minlist
    maxlist=maxlist.tolist(); minlist=minlist.tolist(); B = B.tolist()

    for i in range(Num):
     X[i] = round((X[i] - minlist)/B,2)
    
    return maxlist, minlist, X  

def min_max_2d(X): #(number of data, variable)
    Num=np.shape(X)[0]; variable=np.shape(X)[1]
    maxlist = np.amax(X,axis=0); minlist = np.amin(X,axis=0)
        
    B = maxlist-minlist
    maxlist=maxlist.tolist(); minlist=minlist.tolist(); B = B.tolist() 
    
    for j in range(variable):
     for i in range(Num):
      X[i][j] = round((X[i][j] - minlist[j])/B[j],2)
    
    return maxlist, minlist, X  

###----------------------------------- Random ---------------------------------###
def RanC(new_np, new_head, Rd_stand):
    Random = np.random.choice(len(new_np[1]), Rd_stand)
    #Random = np.arange(0,Rd_stand) 
    print ('Rd :', Random)

    rd_np = np.zeros((len(new_head), Rd_stand))
    print (new_np, type(new_np))
 
    for i in list(range(len(new_head))):
     for k, j in enumerate(Random):
      rd_np[i,k] = new_np[i,j]
    print (rd_np, np.shape(rd_np))     
    return rd_np

###------------------------------------ Plot ----------------------------------###
def heatmap(fcstdata, FC, Case, minv):
     #print (minv)
    fig, ax = plt.subplots(figsize=(10,10)) # all(18,34)
    im = ax.imshow(fcstdata, cmap='bwr', norm=Normalize(vmin=-1, vmax=1)) ## minv (Color value)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(FC)))
    ax.set_yticks(np.arange(len(Case)))

    # ... and label them with the respective list entries
    ax.set_title("Correlation")
    ax.set_xlabel('Variables',fontsize=13)
    ax.set_ylabel('Variables', fontsize=13)
    ax.set_xticklabels(FC, fontsize=11)
    ax.set_yticklabels(Case, fontsize=11)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(Case)):
        for j in range(len(FC)):
          text = ax.text(j, i, round(fcstdata[i,j],2), fontsize=7,  #, fontweight='bold',
                       ha="center", va="center", color="black")

     # ax.set_title("Harvest of local farmers (in tons/year)")
        # orientation='horizontal'-> pad=0.3  or vertical

    # Colorbar 
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cool = fig.colorbar(im, cax=cax)
     #cool.set_label('Correlation', fontsize=13)    
     #cool.set_label('$\Delta$RMSE($^\circ$)', fontsize=13)
    fig.tight_layout()
    plt.savefig('../out/cor.jpg', dpi=600)
    #plt.show()

def importance_plot(header, scores):
    ## Sort
    score_float = sorted(scores.tolist(), key=float, reverse=True)
    #print ("sort", score_float, type(scoreslist), index)
    score_num=list()
    for i in list(range(len(score_float))):
     for j in list(range(len(scores))):
      if (score_float[i] == scores[j]):
        score_num.append(j)
      if (len(scores) == len(score_num)):
        break
    new_header = [ header[i] for i in score_num ]

    # Bar condition
    bar_width=0.15; opacity=0.9

    plt.figure(figsize=(10,9))
    if (len(header) == len(scores)):
     plt.hlines(y=np.nanmean(scores), xmin=0, xmax=len(header)-1, color='m', linestyles='dashed')
    else:
     plt.hlines(y=header[-1], xmin=0, xmax=len(header)-1, color='m', linestyles='dashed')
     del[header[-1]]
    plt.xticks(range(len(new_header)), new_header, rotation=90, fontsize=8)
    plt.yticks(fontsize=10)
    plt.title('Feature_importance')
    plt.xlabel('Variables', fontsize=10, labelpad=5.0)
    plt.ylabel('Scores',fontsize=10, labelpad=6.0)
    plt.scatter(range(len(score_float)), score_float, c='r', marker='o')
    plt.bar(range(len(score_float)), score_float, bar_width, color='green', alpha=opacity, align='center')
    plt.savefig('../out/importance.jpg', dpi=600)
    #plt.show()

def histogram(nwp,obs):
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title('Histogram (nwp)')
    plt.xlabel('Range', fontsize=10, labelpad=5.0)
    plt.ylabel('Number',fontsize=10, labelpad=5.0) 
    plt.hist(nwp, bins=50, color='green', alpha=0.5, label='train')
    plt.hist(obs, bins=50, color='blue', alpha=0.5, label='validation')
#    plt.hist(obs, bins=50, color='blue', alpha=0.5, label='test')
##    plt.hist(nwp+obs, bins=50, color='green', alpha=0.5, label='train')
    plt.legend()
    plt.show()

def pdf_plot(nwp, obs):
    #ax = sns.displot(nwp, kind='kde', color='green', label='Train')
    #ax = sns.displot(obs, kind='kde', color='blue', label='Validation')    
    ax = sns.distplot(nwp, color='green', hist=False, label='Train')
    ax = sns.distplot(obs, color='blue', hist=False, label='Validation')
    plt.legend()
    plt.show()
    
def comparison_plot(nwp, obs):
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title('Relationship (nwp_obs)')
    plt.xlabel('OBS', fontsize=10, labelpad=5.0)
    plt.ylabel('NWP',fontsize=10, labelpad=5.0)
    plt.scatter(obs,nwp,s=15,c='b',marker='o')
    plt.show()

###------------------------------ Data process --------------------------###
## coded (Man-ki, KIM)
def Remove_nan_batch(nwp, obs): # input_size, num_fct, output_size):
    print ('---------- In remove nan batch')
    print ('nwp ' , nwp.shape )
    print ('obs ' , obs.shape )

    variable, day, time = nwp.shape

    #---------------------------------------------------------------------------------------
    # .. Unification of missing data

    print (np.shape(nwp), np.shape(obs), np.shape(nwp[0,:,:]), np.shape(obs[0,:,:]))

    # .... 1. obs to nwp
    print ('Step: obs to nwp')
    for j in range(variable):
       np.place(nwp[j,:,:], obs[0,:,:]==-999., -999.)
       #np.place(nwp[j,:,:], nwp[j,:,:] > 100, 100.)  # humidity

    # .... 2. nwp to obs
    print ('Step: nwp to obs')
    np.place(obs[0,:,:], nwp[0,:,:]==-999., -999.)

    # .... 3. -999 to np.nan
    missing_idx = np.where( nwp == -999. )
    print (missing_idx)
    remove_dim = 1
    print ("missing count = ", len(set(missing_idx[remove_dim])) )

    #print '---------- After unify nan'
    #for i in range(input_size):
    #    print len(np.where(nwp[:,:,i]==-999.)[0])

    np.place(obs, obs==-999., np.nan)
    np.place(nwp, nwp==-999., np.nan)

    #---------------------------------------------------------------------------------------
    # .. Dropping nan

    re_nwp = np.delete( nwp, missing_idx[remove_dim], axis = remove_dim )
    re_obs = np.delete( obs, missing_idx[remove_dim], axis = remove_dim )

    return re_nwp, re_obs

