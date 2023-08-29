import numpy as np
import pandas as pd
from data_split import data_split

import sys
sys.path.insert(0, '../')
from config.global_params import variable_info




def data_load(nwp_file, obs_file, stn_id):
    # 인풋 준비
    nwp_data = np.load(nwp_file)
    obs_data = np.load(obs_file)
    print("="*50, "load data shape")
    print(nwp_data.shape)
    print(obs_data.shape)


    # train([21.01,04, 22.01,04]) / test([23.01,04]) 분할  
    class_split = data_split(nwp_data, obs_data)
    train_nwp, test_nwp, train_obs, test_obs = class_split.get_split_data()
    print("="*50, "split data shape")
    print(train_nwp.shape)
    print(train_obs.shape)
    print(test_nwp.shape)
    print(test_obs.shape)


    # # 결측 가시화
    # import missingno as msno
    # import matplotlib.pyplot as plt
    # msno.matrix(df)



    # 결측제거
    missing_nwp_train = set(np.where(np.isnan(train_nwp))[0])
    missing_obs_train = set(np.where(np.isnan(train_obs))[0])
    missing_all_train = list(missing_nwp_train | missing_obs_train)
    print("결측 합계: ", len(missing_all_train))
    print("결측 index=", missing_all_train)
    dm_nwp_train = np.delete(train_nwp, missing_all_train, 0)
    dm_obs_train = np.delete(train_obs, missing_all_train, 0)
    print("shape of after drop")
    print(dm_nwp_train.shape)
    print(dm_obs_train.shape)

    missing_nwp_test = set(np.where(np.isnan(test_nwp))[0])
    missing_obs_test = set(np.where(np.isnan(test_obs))[0])
    missing_all_test = list(missing_nwp_test | missing_obs_test)
    print("결측 합계: ", len(missing_all_test))
    print("결측 index=", missing_all_test)

    print("test_nwp missing")
    print(test_nwp[missing_all_test,:,0])
    print("test_obs missing")
    print(test_obs[missing_all_test,:,0])
    
    dm_nwp_test = np.delete(test_nwp, missing_all_test, 0)
    dm_obs_test = np.delete(test_obs, missing_all_test, 0)
    print("shape of after drop")
    print(dm_nwp_test.shape)
    print(dm_obs_test.shape)


    # 변수선택
    if stn_id == 875:
        sel_var = ['UGRD_10m', 'VGRD_10m', "TMP_1_5m", 'RH_1_5ma', 'PRMSL_meansealevel', "PRES_surface"]
    else:
        sel_var = ['NDNSW_surface', 'UGRD_10m', 'VGRD_10m', 'RH_1_5ma', 'MAXGUST_0m', 'PRMSL_meansealevel']
        
        
    var_list_dict = list(variable_info.keys())
    var_index = [ var_list_dict.index(i) for i in sel_var ]
    #print(var_list_dict)
    #print(var_index)
    sel_dm_nwp_train = dm_nwp_train[:,1::,var_index]
    sel_dm_nwp_test = dm_nwp_test[:,1::,var_index]
    print("=====================================check")
    print(sel_dm_nwp_test[0,:,1])
    dm_obs_train = dm_obs_train[:,1::,:]
    dm_obs_test = dm_obs_test[:,1::,:]


    print("="*50, "drop data shape")
    print(sel_dm_nwp_train.shape)
    print(dm_obs_train.shape)
    print(sel_dm_nwp_test.shape)
    print(dm_obs_test.shape)

    return sel_dm_nwp_train, sel_dm_nwp_test, dm_obs_train, dm_obs_test