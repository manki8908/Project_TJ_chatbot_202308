def test_data_load_pdate(data_dir, data_per, nwp_element, num_ele, num_stn, num_his, num_fct): 

    # .... load module
    import numpy as np
    import sys
    import os

    sys.path.insert(0, './')
    from test_seqt_read import Squential_read_var
    from test_seqt_read import Squential_read_var_pdate


    # .... initialize
    nwp_dir = data_dir + 'NWP/LC/tran_gmix_'
    obs_dir = data_dir + 'OBS/tran_obs_'

    input_size = num_ele
    output_size = 1

    if nwp_element == 'ALLV': obs_element = 'tmp'


    print nwp_dir
    print nwp_element
    print data_per

    # .... read binary ( nvar, nstn, nhis, nfct )
    nwp_name = nwp_dir + nwp_element + '.' + data_per
    obs_name = obs_dir + obs_element + '.' + data_per

    nwp_exist = os.path.isfile(nwp_name)
    obs_exist = os.path.isfile(obs_name)


    if obs_exist:
        obs_data, ndate = Squential_read_var_pdate(output_size, obs_name, num_stn, num_his, num_fct)
    else:
        sys.exit("STOP Error: Could not found : "+ obs_name)


    if nwp_exist:
        nwp_data = Squential_read_var(input_size,  nwp_name, num_stn, num_his, num_fct)
    else:
        sys.exit("STOP Error: Could not found : "+ nwp_name)

    print "Read obs raw dimension: ", obs_data.shape
    print "Read nwp raw dimension: ", nwp_data.shape


    # .. Rehape for lstm input format
    re_nwp_stn = np.swapaxes(nwp_data,0,3)
    re_obs_stn = np.swapaxes(obs_data,0,3)

    print '---------- Loaded data shape'
    print 'reshape_nwp_stn ' , re_nwp_stn.shape, re_nwp_stn.dtype
    print 'reshape_obs_stn ' , re_obs_stn.shape, re_obs_stn.dtype

    return re_nwp_stn, re_obs_stn, ndate
