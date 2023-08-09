def tran_data_load_nodrop(data_dir, data_per, nwp_element, num_ele, num_stn, num_his, num_fct, run_stn_idx): 

    import numpy as np
    import sys
    import os

    sys.path.insert(0, './inc')
    from test_seqt_read import Squential_read_var
    from test_remv_nann import Remove_nan_batch
    #import random
    from random import sample

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
  
    print ("Read obs raw dimension: "), obs_data.shape
    print ("Read nwp raw dimension: "), nwp_data.shape

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
    #for i in range(input_size):
    #    print len(set(np.where(re_nwp_stn[:,:,i]==-999.)[1]))

    #dropna_nwp, dropna_obs, miss_his = Remove_nan_batch(re_nwp_stn, re_obs_stn, 
    #                                          input_size, num_fct, output_size)


    #print ('---------- Shape of after drop nan ')
    #print (dropna_nwp.shape)
    #print (dropna_obs.shape)


    ## .... devide dataset ( nvar, ntrd/nevl, nfct )
    #uni_his = dropna_nwp.shape[1]  
    #num_evl = int(round(uni_his*0.2))
    #eval_idx = sample( range(uni_his), num_evl )

    #tran_x = np.delete( dropna_nwp, eval_idx, axis=1 )
    #tran_y = np.delete( dropna_obs, eval_idx, axis=1 )

    #eval_x, eval_y = [], []

    #for i in eval_idx:
    #    eval_x.append(dropna_nwp[:,i,:])
    #    eval_y.append(dropna_obs[:,i,:])
    #eval_x = np.array(eval_x)
    #eval_y = np.array(eval_y)
    #eval_x = np.swapaxes(eval_x,0,1)
    #eval_y = np.swapaxes(eval_y,0,1)

    #print '---------- Devided data shape'
    #print 'tran_x : ', tran_x.shape
    #print 'tran_y : ', tran_y.shape
    #print 'eval_x : ', eval_x.shape
    #print 'eval_y : ', eval_y.shape

    #return tran_x, tran_y, eval_x, eval_y, tran_x.shape[1], eval_x.shape[1]
    #return dropna_nwp, dropna_obs, miss_his
    return re_nwp_stn, re_obs_stn
