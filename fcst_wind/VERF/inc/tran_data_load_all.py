def tran_data_load_all(data_dir, data_per, nwp_element, num_ele, num_stn, num_his, num_fct): 

    import numpy as np
    import sys
    import os

    sys.path.insert(0, './inc')
    from test_seqt_read import Squential_read_var
    from test_remv_nann import Remove_nan_batch

    # .... initialize

    nwp_dir = data_dir + 'NWP/LC/tran_gmix_'
    #nwp_dir = data_dir + 'NWP/LC_V2/tran_gmix_'   # nppm test for station(IGS data)
    obs_dir = data_dir + 'OBS/tran_obs_'

    input_size = num_ele
    output_size = 1

    print (nwp_element, num_ele)

    if num_ele == 74:
       nwp_element == 'ALLV'
       obs_element = 'tmp'
    if num_ele == 7:
       nwp_element = 'ALLV_nvar07'
       obs_element = 'tmp'


    # .... read binary ( nvar, nstn, nhis, nfct )
    #print ( type(nwp_dir) )
    #print ( type(nwp_element) )
    #print ( type(data_per) )
    nwp_name = nwp_dir + nwp_element + '.' + data_per
    obs_name = obs_dir + obs_element + '.' + data_per
    
    nwp_exist = os.path.isfile(nwp_name)
    obs_exist = os.path.isfile(obs_name)

    if obs_exist:
        obs_data = Squential_read_var(output_size, obs_name, num_stn, num_his, num_fct)
    else:
        sys.exit("STOP Error: Could not found : "+ obs_name)

    if nwp_exist:
        nwp_data = Squential_read_var(input_size,  nwp_name, num_stn, num_his, num_fct)
    else:
        sys.exit("STOP Error: Could not found : "+ nwp_name)
  
    print ("Read obs raw dimension: ", obs_data.shape)
    print ("Read nwp raw dimension: ", nwp_data.shape)


    #.. Check
    #for i in range(input_size):
    #    print len(set(np.where(re_nwp_stn[:,:,i]==-999.)[1]))

    #dropna_nwp, dropna_obs, miss_his = Remove_nan_batch(re_nwp_stn, re_obs_stn, 
    #                                          input_size, num_fct, output_size)


    #print ('---------- Shape of after drop nan ')
    #print (dropna_nwp.shape)
    #print (dropna_obs.shape)


    return nwp_data, obs_data
    #return dropna_nwp, dropna_obs, miss_his


