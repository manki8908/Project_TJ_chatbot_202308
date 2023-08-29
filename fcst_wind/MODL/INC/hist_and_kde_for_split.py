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

#.. local
import sys
sys.path.insert(0, './')
from plot_histogram import plot_histogram
from plot_kde import plot_kde_tran_vald_test, plot_kde, plot_kde_tran_vald



#def hist_and_kde_for_split_UV(exp_name, tran_peri, run_stn_id, prt_outdir, tran_y, vald_y, test_y):
def hist_and_kde_for_split_UV(exp_name, tran_peri, run_stn_id, prt_outdir, tran_x, vald_x, test_x, tran_y, vald_y, test_y):

    print("plotting KDE .. " )
    print("tran_y shape: ", tran_y.shape)
    print("vald_y shape: ", vald_y.shape)
    print("test_y shape: ", test_y.shape)


    # .. kde
    save_out_name = prt_outdir + "KDE_ecmw_tran_vald_nwp_uuu" + tran_peri + "_" + str(run_stn_id)
    plot_kde_tran_vald_test("TRAN_NWP", "VALD_NWP", "TEST_NWP", tran_x[:,:,0:1], vald_x[:,:,0:1], test_x[:,:,0:1], run_stn_id, tran_peri, save_out_name)
    save_out_name = prt_outdir + "KDE_ecmw_tran_vald_nwp_vvv" + tran_peri + "_" + str(run_stn_id)
    plot_kde_tran_vald_test("TRAN_NWP", "VALD_NWP", "TEST_NWP", tran_x[:,:,1:2], vald_x[:,:,1:2], test_x[:,:,1:2], run_stn_id, tran_peri, save_out_name)

    save_out_name = prt_outdir + "KDE_ecmw_tran_vald_obs_uuu" + tran_peri + "_" + str(run_stn_id)
    plot_kde_tran_vald_test("TRAN_OBS", "VALD_OBS", "TEST_OBS", tran_y[:,:,0:1], vald_y[:,:,0:1], test_y[:,:,0:1], run_stn_id, tran_peri, save_out_name)
    save_out_name = prt_outdir + "KDE_ecmw_tran_vald_obs_vvv" + tran_peri + "_" + str(run_stn_id)
    plot_kde_tran_vald_test("TRAN_OBS", "VALD_OBS", "TEST_OBS", tran_y[:,:,1:2], vald_y[:,:,1:2], test_y[:,:,1:2], run_stn_id, tran_peri, save_out_name)



def hist_and_kde_for_split_spd(exp_name, tran_peri, run_stn_id, prt_outdir, tran_y, vald_y, test_y):

    print("plotting KDE .. " )
    print("tran_y shape: ", tran_y.shape)
    print("vald_y shape: ", vald_y.shape)
    print("test_y shape: ", test_y.shape)


    # .. kde
    save_out_name = prt_outdir + "KDE_ecmw_tran_vald_obs_spd" + tran_peri + "_" + str(run_stn_id)
    plot_kde_tran_vald_test("TRAN_OBS", "VALD_OBS", "TEST_OBS", tran_y[:,:,0:1], vald_y[:,:,0:1], test_y[:,:,0:1], run_stn_id, tran_peri, save_out_name)



def hist_and_kde_for_split(var_name, exp_name, tran_peri, run_stn_id, prt_outdir, dropna_tran_x, dropna_tran_y, dropna_vald_x, dropna_vald_y):

    # .. kde
    save_out_name = prt_outdir + "KDE_ecmw_tran_var6_" + tran_peri + "_" + str(run_stn_id)
    plot_kde("TRAN_NWP", "TRAN_OBS", dropna_tran_x, dropna_tran_y, run_stn_id, tran_peri, save_out_name)
    save_out_name = prt_outdir + "KDE_ecmw_vald_var6_" + tran_peri + "_" + str(run_stn_id)
    plot_kde("VALD_NWP", "VALD_OBS", dropna_vald_x, dropna_vald_y, run_stn_id, tran_peri, save_out_name)
    
    save_out_name = prt_outdir + "KDE_ecmw_tran_vald_nwp_" + tran_peri + "_" + str(run_stn_id)
    plot_kde_tran_vald("TRAN_NWP", "VALD_NWP", dropna_tran_x, dropna_vald_x, run_stn_id, tran_peri, save_out_name)
    save_out_name = prt_outdir + "KDE_ecmw_tran_vald_obs_" + tran_peri + "_" + str(run_stn_id)
    plot_kde_tran_vald("TRAN_OBS", "VALD_OBS", dropna_tran_y, dropna_vald_y, run_stn_id, tran_peri, save_out_name)
 
