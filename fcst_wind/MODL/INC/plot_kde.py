#.. module
import os
import numpy as np
import scipy
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
import copy

def plot_kde_tran_vald_test(label1, label2, label3, tran_obs, vald_obs, test_obs, stn_id, tran_peri, save_out_name):

   print ("tran_obs shape:" ,tran_obs.shape)
   print ("vald_obs shape:" ,vald_obs.shape)
   print ("test_obs shape:" ,test_obs.shape)

   tran_obs = tran_obs[:,:,0:1]
   tran_obs = tran_obs.ravel()
   vald_obs = vald_obs[:,:,0:1]
   vald_obs = vald_obs.ravel()
   test_obs = test_obs[:,:,0:1]
   test_obs = test_obs.ravel()


   ax1 = sns.kdeplot(tran_obs, legend=True, color='red',
                    label=f'{label1}, sample size={tran_obs.shape[0]}')
   tnwp_x, tnwp_y = ax1.get_lines()[0].get_data()
   ax2 = sns.kdeplot(vald_obs, legend=True, color='lime', alpha=0.7,
                    label=f'{label2}, sample size={vald_obs.shape[0]}')
   tobs_x, tobs_y = ax2.get_lines()[1].get_data()
   ax3 = sns.kdeplot(test_obs, legend=True, color='black', alpha=0.7,
                    label=f'{label3}, sample size={test_obs.shape[0]}')
   tobs_x, tobs_y = ax3.get_lines()[2].get_data()

   ax3.set_title(f"KDE plot {tran_peri}", loc='left', fontsize=9, fontweight='bold')
   ax3.set_title(f"STN_ID: {stn_id}", loc='right', fontsize=9, fontweight='bold')
   #ax3.set_xlabel("Temeprature(\u00B0C)")
   ax3.set_xlabel("Wind vector(m/s)")
   #ax3.set_xlim(left=-15, right=15)  # temperature
   ax3.set_xlim(left=-0.5, right=1.5)  # sacled wind vector
   #ax3.set_ylim(-0.001, 0.4)
   ax3.set_ylim(-0.001, 5.)
   ax3.legend(loc='upper left', fontsize=8, frameon=True, facecolor='white')

   ## .. Calc. median
   #tnwp_cdf = scipy.integrate.cumtrapz(tnwp_y, tnwp_x, initial=0)
   #tobs_cdf = scipy.integrate.cumtrapz(tobs_y, tobs_x, initial=0)

   #tnwp_near_05 = np.abs(tnwp_cdf-0.5).argmin()
   #tobs_near_05 = np.abs(tobs_cdf-0.5).argmin()

   #tnwp_x_median, tnwp_y_median = tnwp_x[tnwp_near_05], tnwp_y[tnwp_near_05]
   #tobs_x_median, tobs_y_median = tobs_x[tobs_near_05], tobs_y[tobs_near_05]

   #print ( "calc from scipy")
   #print ( f'tnwp_x_median: {tnwp_x_median}' )
   #print ( f'tobs_x_median: {tobs_x_median}' )

   print ( "calc from numpy")
   tran_x_median = np.median(tran_obs)
   vald_x_median = np.median(vald_obs)
   test_x_median = np.median(test_obs)
   print ( f'tran_x_median: {tran_x_median}' )
   print ( f'vald_x_median: {vald_x_median}' )
   print ( f'test_x_median: {test_x_median}' )

   mu_tran = tran_obs.mean()
   min_tran = tran_obs.min()
   max_tran = tran_obs.max()
   mu_vald = vald_obs.mean()
   min_vald = vald_obs.min()
   max_vald = vald_obs.max()
   mu_test = test_obs.mean()
   min_test = test_obs.min()
   max_test = test_obs.max()
   
   textstr = '\n'.join((
              r'<tran data>',
              r'$\mu=%.2f$'%(mu_tran,),
              r'$  min=%.2f$'%(min_tran,),
              r'$  max=%.2f$'%(max_tran,),
              r'<valid data>',
              r'$\mu=%.2f$'%(mu_vald,),
              r'$  min=%.2f$'%(min_vald,),
              r'$  max=%.2f$'%(max_vald,),
              r'<test data>',
              r'$\mu=%.2f$'%(mu_test,),
              r'$  min=%.2f$'%(min_test,),
              r'$  max=%.2f$'%(max_test,)
               ))

   props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
   ax1.text(0.8,0.95, textstr, transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=props)

   ax1.vlines(tran_x_median, 0, 9, color='red', linestyle='--', linewidth=0.6)
   ax2.vlines(vald_x_median, 0, 9, color='lime', linestyle='--', linewidth=0.6)
   ax3.vlines(test_x_median, 0, 9, color='black', linestyle='--', linewidth=0.6)

   # .. save fig
   print(save_out_name)
   plt.savefig(save_out_name)

   plt.clf()



def plot_kde(label1, label2, dropna_nwp, dropna_obs, stn_id, tran_peri, save_out_name):

   print ("dropna_nwp shape:" ,dropna_nwp.shape)
   print ("dropna_obs shape:" ,dropna_obs.shape)
   
   dropna_nwp = dropna_nwp[:,:,0:1]
   dropna_nwp = dropna_nwp.ravel()
   dropna_obs = dropna_obs.ravel()
   
   
   ax1 = sns.kdeplot(dropna_nwp, legend=True, color='red',
                    label=f'{label1}, sample size={dropna_nwp.shape[0]}')
   tnwp_x, tnwp_y = ax1.get_lines()[0].get_data()
   ax2 = sns.kdeplot(dropna_obs, legend=True, color='lime', alpha=0.7,
                    label=f'{label2}, sample size={dropna_obs.shape[0]}')
   tobs_x, tobs_y = ax2.get_lines()[1].get_data()
   
   ax2.set_title(f"KDE plot {tran_peri}", loc='left', fontsize=9, fontweight='bold')
   ax2.set_title(f"STN_ID: {stn_id}", loc='right', fontsize=9, fontweight='bold')
   ax2.set_xlabel("Temeprature(\u00B0C)")
   ax2.set_xlim(left=-25, right=45)
   ax2.set_ylim(-0.001, 0.06)
   ax2.legend(loc='upper left', fontsize=8, frameon=True, facecolor='white')
   
   # .. Calc. median
   #tnwp_cdf = scipy.integrate.cumtrapz(tnwp_y, tnwp_x, initial=0)
   #tobs_cdf = scipy.integrate.cumtrapz(tobs_y, tobs_x, initial=0)
   #
   #tnwp_near_05 = np.abs(tnwp_cdf-0.5).argmin()
   #tobs_near_05 = np.abs(tobs_cdf-0.5).argmin()
   #
   #tnwp_x_median, tnwp_y_median = tnwp_x[tnwp_near_05], tnwp_y[tnwp_near_05]
   #tobs_x_median, tobs_y_median = tobs_x[tobs_near_05], tobs_y[tobs_near_05]
   #
   #print ( "calc from scipy")
   #print ( f'tnwp_x_median: {tnwp_x_median}' )
   #print ( f'tobs_x_median: {tobs_x_median}' )

   print ( "calc from numpy")
   tnwp_x_median = np.median(dropna_nwp)
   tobs_x_median = np.median(dropna_obs)
   print ( f'tnwp_x_median: {tnwp_x_median}' )
   print ( f'tobs_x_median: {tobs_x_median}' )
   
   ax1.vlines(tnwp_x_median, 0, 0.06, color='red', linestyle='--', linewidth=0.6)
   ax2.vlines(tobs_x_median, 0, 0.06, color='blue', linestyle='--', linewidth=0.6)
  
   mu_tran = dropna_nwp.mean()
   min_tran = dropna_nwp.min()
   max_tran = dropna_nwp.max()
   mu_vald = dropna_obs.mean()
   min_vald = dropna_obs.min()
   max_vald = dropna_obs.max()

   textstr = '\n'.join((
              r'<nwp>',
              r'$\mu=%.2f$'%(mu_tran,),
              r'$  min=%.2f$'%(min_tran,),
              r'$  max=%.2f$'%(max_tran,),
              r'<obs>',
              r'$\mu=%.2f$'%(mu_vald,),
              r'$  min=%.2f$'%(min_vald,),
              r'$  max=%.2f$'%(max_vald,)
               ))

   props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
   ax1.text(0.8,0.95, textstr, transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=props)


 
   # .. save fig
   print(save_out_name)
   plt.savefig(save_out_name)

   plt.clf()




def plot_kde_tran_vald(label1, label2, dropna_nwp, dropna_obs, stn_id, tran_peri, save_out_name):

   print ("dropna_nwp shape:" ,dropna_nwp.shape)
   print ("dropna_obs shape:" ,dropna_obs.shape)

   dropna_nwp = dropna_nwp[:,:,0:1]
   dropna_nwp = dropna_nwp.ravel()
   dropna_obs = dropna_obs[:,:,0:1]
   dropna_obs = dropna_obs.ravel()


   ax1 = sns.kdeplot(dropna_nwp, legend=True, color='red',
                    label=f'{label1}, sample size={dropna_nwp.shape[0]}')
   tnwp_x, tnwp_y = ax1.get_lines()[0].get_data()
   ax2 = sns.kdeplot(dropna_obs, legend=True, color='lime', alpha=0.7,
                    label=f'{label2}, sample size={dropna_obs.shape[0]}')
   tobs_x, tobs_y = ax2.get_lines()[1].get_data()

   ax2.set_title(f"KDE plot {tran_peri}", loc='left', fontsize=9, fontweight='bold')
   ax2.set_title(f"STN_ID: {stn_id}", loc='right', fontsize=9, fontweight='bold')
   ax2.set_xlabel("Temeprature(\u00B0C)")
   ax2.set_xlim(left=-25, right=45)
   ax2.set_ylim(-0.001, 0.06)
   ax2.legend(loc='upper left', fontsize=8, frameon=True, facecolor='white')

   ## .. Calc. median
   #tnwp_cdf = scipy.integrate.cumtrapz(tnwp_y, tnwp_x, initial=0)
   #tobs_cdf = scipy.integrate.cumtrapz(tobs_y, tobs_x, initial=0)

   #tnwp_near_05 = np.abs(tnwp_cdf-0.5).argmin()
   #tobs_near_05 = np.abs(tobs_cdf-0.5).argmin()

   #tnwp_x_median, tnwp_y_median = tnwp_x[tnwp_near_05], tnwp_y[tnwp_near_05]
   #tobs_x_median, tobs_y_median = tobs_x[tobs_near_05], tobs_y[tobs_near_05]

   #print ( "calc from scipy")
   #print ( f'tnwp_x_median: {tnwp_x_median}' )
   #print ( f'tobs_x_median: {tobs_x_median}' )

   print ( "calc from numpy")
   tnwp_x_median = np.median(dropna_nwp)
   tobs_x_median = np.median(dropna_obs)
   print ( f'tnwp_x_median: {tnwp_x_median}' )
   print ( f'tobs_x_median: {tobs_x_median}' )

   mu_tran = dropna_nwp.mean()
   min_tran = dropna_nwp.min()
   max_tran = dropna_nwp.max()
   mu_vald = dropna_obs.mean()
   min_vald = dropna_obs.min()
   max_vald = dropna_obs.max()
   
   textstr = '\n'.join((
              r'<tran data>',
              r'$\mu=%.2f$'%(mu_tran,),
              r'$  min=%.2f$'%(min_tran,),
              r'$  max=%.2f$'%(max_tran,),
              r'<valid data>',
              r'$\mu=%.2f$'%(mu_vald,),
              r'$  min=%.2f$'%(min_vald,),
              r'$  max=%.2f$'%(max_vald,)
               ))

   props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
   ax1.text(0.8,0.95, textstr, transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=props)

   ax1.vlines(tnwp_x_median, 0, 0.06, color='red', linestyle='--', linewidth=0.6)
   ax2.vlines(tobs_x_median, 0, 0.06, color='blue', linestyle='--', linewidth=0.6)

   # .. save fig
   print(save_out_name)
   plt.savefig(save_out_name)

   plt.clf()