# .. 스케일링 및 데이터 분할
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from step_sampling_for_date import step_sampling_for_date
from hist_and_kde_for_split import hist_and_kde_for_split, hist_and_kde_for_split_UV




def data_scaling(output_size, tran_rate, nbin, random_seed, exp_name, sel_dm_nwp_train, sel_dm_nwp_test, dm_obs_train, dm_obs_test):


    #-------------------------------------------------------------------------
    # .. Normalize

    # .. initialaize
    tr_b, tr_s, tr_f = sel_dm_nwp_train.shape[0], sel_dm_nwp_train.shape[1], sel_dm_nwp_train.shape[2]      
    ts_b, ts_s, ts_f = sel_dm_nwp_test.shape[0], sel_dm_nwp_test.shape[1], sel_dm_nwp_test.shape[2]      

    # .. get restorator with obs range
    nwp_scaler = MinMaxScaler()   # copy default true
    obs_scaler = MinMaxScaler()
    nwp_scaler.fit(sel_dm_nwp_train.view().reshape(tr_b*tr_s, tr_f))
    obs_scaler.fit(dm_obs_train.view().reshape(tr_b*tr_s, output_size))

    # .. feature normalize   ( train seq, feature = test seq, feature )
    nor_dm_nwp_train = nwp_scaler.transform(sel_dm_nwp_train.reshape(tr_b*tr_s, tr_f))
    nor_dm_nwp_train = nor_dm_nwp_train.reshape(tr_b,tr_s,tr_f)
    nor_dm_nwp_test = nwp_scaler.transform(sel_dm_nwp_test.reshape(ts_b*ts_s, ts_f))
    nor_dm_nwp_test = nor_dm_nwp_test.reshape(ts_b,ts_s,ts_f)

    nor_dm_obs_train = obs_scaler.transform(dm_obs_train.reshape(tr_b*tr_s, output_size))
    nor_dm_obs_train = nor_dm_obs_train.reshape(tr_b,tr_s, output_size)
    nor_dm_obs_test = obs_scaler.transform(dm_obs_test.reshape(ts_b*ts_s, output_size))
    nor_dm_obs_test = nor_dm_obs_test.reshape(ts_b,ts_s, output_size)


    print ('---------- Final training data shape')
    print(type(nor_dm_nwp_train))
    print ('tran nwp : ', nor_dm_nwp_train.shape)
    print ('tran obs : ', nor_dm_obs_train.shape)
    print ('test nwp : ', nor_dm_nwp_test.shape)
    print ('test obs : ', nor_dm_obs_test.shape)



    return nor_dm_nwp_train,nor_dm_nwp_train,nor_dm_nwp_test,nor_dm_nwp_test,nor_dm_obs_train,nor_dm_obs_train,nor_dm_obs_test,nor_dm_obs_test

