def best_data_load(data_dir, data_per, element, stn_idx, num_stn, num_his, num_fct): 

    # .... load module
    import numpy as np
    import sys
    sys.path.insert(0, '/s4/home/nimr/mankicom/STUDY/DEV_SHRT_G128/EXPR_VERF/VERF_G128_LSTM_47108_XNOR/inc')
    from test_seqt_read import Squential_read_var
    from test_remv_nann import Remove_nan_batch
    from random import * 
    import torch


    # .... initialize
    mos_name = []

    mos_dir = data_dir + 'NWP/BEST/tran_best_'

    print element

    if element[0] == 'TMP': mos_element = 'T3H' 

    input_size = len(element)
    output_size = 1



    # .... read binary ( nvar, nstn, nhis, nfct )
    mos_name.append( mos_dir + mos_element + '.' + data_per )

    print "mos_name", mos_name[0]
    print num_stn, num_his, num_fct

    mos_data = Squential_read_var(output_size, mos_name[0], num_stn, num_his, num_fct)
 


    # .... for one station ( nvar, nstn, nhis, nfct ) --> (nvar, nhis, nfct)
    # .. 37 is 47108
    mos_data = mos_data[:,stn_idx,:,:]

    print 'print read pmos data shape= ', mos_data.shape

    reshape_pmos_x = np.swapaxes(mos_data,0,2)

    print 'print reshape_pmos_y ' , reshape_pmos_x.shape
#    print 'print test_x ' , test_x.shape
#    print 'print test_y ' , test_y.shape

    #dropna_pmos_x, dropna_test_x = Remove_nan_batch(reshape_pmos_x, test_x, 
    #                                                    input_size, num_fct, output_size)
    #dropna_pmos_x, dropna_test_y = Remove_nan_batch(reshape_pmos_x, test_y, 
    #                                                    input_size, num_fct, output_size)


#    print 'print dropna_x ' , test_x.shape
#    print 'print dropna_y ' , test_y.shape
#    print 'print dropna_p ' , dropna_pmos_x.shape

    return reshape_pmos_x
    #return test_x, test_y, reshape_pmos_x
