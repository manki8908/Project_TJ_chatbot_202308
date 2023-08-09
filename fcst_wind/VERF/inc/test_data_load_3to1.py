def test_data_load_3to1(data_dir, data_per, nwp_element, num_ele, num_his, in_num_fct, out_num_fct, run_stn_id): 

    import numpy as np
    import sys
    import os

    sys.path.insert(0, './inc')
    #from test_seqt_read import Squential_read_var
    from test_read_fortranfile import Read_tran_fortranfile
    from test_remv_nann_3to1 import Remove_nan_batch_3to1
    from test_find_stnidx import find_stn_idx

    # .... initialize

    nwp_dir = data_dir + 'NWP/LC/tran_gmix_'
    obs_dir = data_dir + 'OBS/tran_obs_'

    input_size = num_ele
    output_size = 1

    if num_ele == 70:
       nwp_element == 'ALLV'
       obs_element = 'tmp'
    if num_ele == 7:
       nwp_element = 'ALLV_nvar07'
       obs_element = 'tmp'


    # .... read binary ( nvar, nstn, nhis, nfct )
    nwp_name = nwp_dir + nwp_element + '.' + data_per + '_' + '%05d'%run_stn_id
    obs_name = obs_dir + obs_element + '.' + data_per + '_' + '%05d'%run_stn_id
    
    nwp_exist = os.path.isfile(nwp_name)
    obs_exist = os.path.isfile(obs_name)

    if obs_exist:
        #obs_data, obs_stn_list = Squential_read_var(output_size, obs_name, num_his, num_fct)
        obs_data, obs_stn_list = Read_tran_fortranfile(obs_name, output_size, num_his, out_num_fct)
    else:
        sys.exit("STOP Error: Could not found : "+ obs_name)

    if nwp_exist:
        #nwp_data, nwp_stn_list = Squential_read_var(input_size,  nwp_name, num_his, num_fct)
        nwp_data, nwp_stn_list = Read_tran_fortranfile(nwp_name, input_size, num_his, in_num_fct)
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

    # .... for one station (nvar, nstn, nhis, nfct) - >  ( nvar, nhis, nfct )
    nwp_stn = nwp_data[:,run_stn_idx,:,:]
    obs_stn = obs_data[:,run_stn_idx,:,:]


    # .. Rehape for lstm input format
    re_nwp_stn = np.swapaxes(nwp_stn,0,2)
    re_obs_stn = np.swapaxes(obs_stn,0,2)
    
    print ('---------- Reshape for lstm input dimension')
    print ('reshape_nwp_stn ' , re_nwp_stn.shape )
    print ('reshape_obs_stn ' , re_obs_stn.shape )

    #.. Check
    np.place(re_obs_stn, re_obs_stn==-999., np.nan)
    np.place(re_nwp_stn, re_nwp_stn==-999., np.nan)

    #.. Check
    #for i in range(input_size):
    #    print len(set(np.where(re_nwp_stn[:,:,i]==-999.)[1]))

    #dropna_nwp, dropna_obs = Remove_nan_batch_3to1(re_nwp_stn, re_obs_stn, 
    #                                 input_size, output_size)


    #print ('---------- Shape of after drop nan ')
    #print (dropna_nwp.shape)
    #print (dropna_obs.shape)


    #return dropna_nwp, dropna_obs
    return re_nwp_stn, re_obs_stn
