#.. module
import os
import numpy as np
import pandas as pd
import copy
import sys

def bias_rmse(nwp, obs, stn_id, nwp_name):

        num_mon = obs.shape[0]
        num_his = obs.shape[1]
        num_fct = obs.shape[2]
        num_stn = obs.shape[3]

        nwp_stn_bias = np.ndarray( shape=(num_mon,num_fct,num_stn), dtype=np.float )
        nwp_stn_rmse = np.ndarray( shape=(num_mon,num_fct,num_stn), dtype=np.float )
        nwp_stn_bias.fill(np.nan)
        nwp_stn_rmse.fill(np.nan)

        for i in range(num_stn):

                for j in range(num_mon):

                        #print (f'------- {stn_id[i]}station/{j}month/{nwp_name} cal rmse ---------------')

                        nwp_stn = copy.deepcopy( nwp[j,:,:,i] )
                        obs_stn = copy.deepcopy( obs[j,:,:,i] )
                        #nwp_stn = nwp[j,:,:,i]
                        #obs_stn = obs[j,:,:,i]

                        # .. unify nan
                        np.place( nwp_stn, np.isnan(obs_stn), np.nan )
                        np.place( obs_stn, np.isnan(nwp_stn), np.nan )


                        # .. Remove missing value for num_his dimension
                        mis_his_all = np.where( np.isnan(nwp_stn) )
                        #print ( 'mis_his count:', len(set(mis_his_all[0])) )

                        if len(set(mis_his_all[0])) < 28:
                                #print ('... Remove missing')
                                flt_nwp_stn = np.delete( nwp_stn, mis_his_all[0], axis = 0 )
                                flt_obs_stn = np.delete( obs_stn, mis_his_all[0], axis = 0 )
                                #print (flt_nwp_stn.shape)
                                #print (flt_obs_stn.shape)

                                nwp_stn_bias[j,:,i] = np.mean( flt_nwp_stn - flt_obs_stn, axis=0 )
                                nwp_stn_rmse[j,:,i] = np.sqrt( ((flt_nwp_stn - flt_obs_stn )**2).mean(axis=0) )


        return nwp_stn_bias, nwp_stn_rmse

