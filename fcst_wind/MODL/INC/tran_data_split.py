def tran_data_split(data_dir, data_per, nwp_element, input_size, output_size, num_his, num_fct, run_stn_id, tran_rate, nbin, random_seed): 

    import numpy as np
    import sys
    import os

    sys.path.insert(0, './inc')
    #from test_seqt_read import Squential_read_var
    from test_read_fortranfile import Read_tran_fortranfile
    #from test_remv_nann import Remove_nan_batch
    from test_remv_nann import Remove_nan_batch_first
    from test_find_stnidx import find_stn_idx
    #from data_split_5years import data_split_5years
    from step_sampling_for_date import step_sampling_for_date

    # .... initialize
    #nwp_dir = data_dir + 'tran_emix_'
    #obs_dir = data_dir + 'tran_obs_'
    nwp_dir = data_dir + 'NWP/LC/tran_gmix_'
    obs_dir = data_dir + 'OBS/tran_obs_'

    nwp_element = 'ALLV_nvar'+str(input_size)   # for 3to1 data 
    #obs_element = 'uvw'
    obs_element = 'spd'


    # .... read binary ( nvar, nstn, nhis, nfct )
    nwp_name = nwp_dir + nwp_element + '.' + data_per + '_' + '%05d'%run_stn_id
    obs_name = obs_dir + obs_element + '.' + data_per + '_' + '%05d'%run_stn_id
    
    nwp_exist = os.path.isfile(nwp_name)
    obs_exist = os.path.isfile(obs_name)

    if obs_exist:
        #obs_data, obs_stn_list = Squential_read_var(output_size, obs_name, num_his, num_fct)
        obs_data, obs_stn_list = Read_tran_fortranfile(obs_name, output_size, num_his, num_fct)
    else:
        sys.exit("STOP Error: Could not found : "+ obs_name)

    if nwp_exist:
        #nwp_data, nwp_stn_list = Squential_read_var(input_size,  nwp_name, num_his, num_fct)
        nwp_data, nwp_stn_list = Read_tran_fortranfile(nwp_name, input_size, num_his, num_fct)
    else:
        sys.exit("STOP Error: Could not found : "+ nwp_name)
  
    print ("Read obs raw dimension: "), obs_data.shape
    print ("Read nwp raw dimension: "), nwp_data.shape

    # .. check station dimension pair
    if ( len(nwp_stn_list) != len(obs_stn_list) ):
       sys.exit("STOP Error: len(nwp_stn_list) != len(obs_stn_list)") 
    if len( set(nwp_stn_list)^set(obs_stn_list) ) > 0:
       sys.exit("set(nwp_stn_list)^set(obs_stn_list) ) > 0")


    # .. Find stn_id_idx 
    run_stn_idx = find_stn_idx(obs_stn_list, run_stn_id)
    print("Find run_stn_idx{} from data".format(run_stn_idx))

    # .... for one station (nvar, nstn, nhis, nfct) - >  ( nvar, nhis, nfct )
    nwp_stn = nwp_data[:,run_stn_idx,:,:]
    obs_stn = obs_data[:,run_stn_idx,:,:]


    # .. Rehape for lstm input format       (seq, batch, feature)
    re_nwp_stn = np.swapaxes(nwp_stn,0,2)
    re_obs_stn = np.swapaxes(obs_stn,0,2)
    re_nwp_stn = np.swapaxes(re_nwp_stn,0,1)
    re_obs_stn = np.swapaxes(re_obs_stn,0,1)
    
    print ('---------- Reshape for lstm input dimension')
    print ('reshape_nwp_stn ' , re_nwp_stn.shape )
    print ('reshape_obs_stn ' , re_obs_stn.shape )


    # .. split tran/vald for 5years data( 2016.01 ~ 2020.12 )
    #tran_x, tran_y, vald_x, vald_y = data_split_5years(re_nwp_stn, re_obs_stn, tran_rate, random_seed) 
    tran_x, tran_y, vald_x, vald_y = step_sampling_for_date(re_nwp_stn, re_obs_stn, tran_rate, nbin, random_seed)


    #.. Check
    #for i in range(input_size):
    #    print len(set(np.where(re_nwp_stn[:,:,i]==-999.)[1]))

    dropna_tran_x, dropna_tran_y = Remove_nan_batch_first(tran_x, tran_y,
                                                    input_size, num_fct, output_size)
    dropna_vald_x, dropna_vald_y = Remove_nan_batch_first(vald_x, vald_y,
                                                    input_size, num_fct, output_size)

    print ('---------- Shape of after drop nan ')
    print ('tran x: ', dropna_tran_x.shape)
    print ('tran y: ', dropna_tran_y.shape)
    print ('vald x: ', dropna_vald_x.shape)
    print ('vald y: ', dropna_vald_y.shape)


    return dropna_tran_x, dropna_tran_y, dropna_vald_x, dropna_vald_y
