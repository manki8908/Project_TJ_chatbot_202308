def Remove_nan_batch(nwp, obs, input_size, num_fct, output_size):

    import numpy as np
    import pandas as pd


    print ('---------- In remove nan batch')
    print ('nwp ' , nwp.shape )
    print ('obs ' , obs.shape )

    #---------------------------------------------------------------------------------------
    # .. Unification of missing data

    # .... 1. obs to nwp
    for i in range(input_size):
        np.place(nwp[:,:,i], obs[:,:,0]==-999., -999.)

    # .... 2. nwp to obs
    np.place(obs[:,:,0], nwp[:,:,0]==-999., -999.)

    # .... 3. -999 to np.nan
    missing_idx = np.where( nwp == -999. )
    remove_dim = 1
    #print ("missing: ", missing_idx)

    #print '---------- After unify nan'
    #for i in range(input_size):
    #    print len(np.where(nwp[:,:,i]==-999.)[0])

    np.place(obs, obs==-999., np.nan)
    np.place(nwp, nwp==-999., np.nan)

    #---------------------------------------------------------------------------------------
    # .. Dropping nan

    re_nwp = nwp
    re_obs = obs

    print ("missing count = ", len(set(missing_idx[remove_dim])) )
    print ("missing days = ", missing_idx[remove_dim])
    if len(set(missing_idx[remove_dim])) != nwp.shape[remove_dim]:
       print (f'Remove nan {missing_idx}')
       re_nwp = np.delete( nwp, missing_idx[remove_dim], axis = remove_dim )
       re_obs = np.delete( obs, missing_idx[remove_dim], axis = remove_dim )

    #return re_nwp, re_obs, missing_idx
    return re_nwp, re_obs
