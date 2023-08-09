def pmos_data_load_all(data_dir, data_per, mos_name, element, num_stn, num_his, num_fct): 

    # .... load module
    import numpy as np
    import sys
    sys.path.insert(0, './inc')
    from test_seqt_read import Squential_read_var
    from test_remv_nann import Remove_nan_batch


    # .... initialize
    mos_file = []

    if mos_name == 'pmos': mos_dir = data_dir + 'NWP/PMOS/tran_pmos_'
    if mos_name == 'emos': mos_dir = data_dir + 'NWP/ECMW/tran_ecmw_'
    if mos_name == 'ecnp': mos_dir = data_dir + 'NWP/ECNP/tran_ecmw_'
    if mos_name == 'best': mos_dir = data_dir + 'NWP/BEST/tran_best_'

    print (f'MOS {element} read')

    if element[0] == 'TMP': mos_element = 'T3H' 

    input_size = len(element)
    output_size = 1



    # .... read binary ( nvar, nstn, nhis, nfct )
    mos_file.append( mos_dir + mos_element + '.' + data_per )

    print ("mos_file", mos_file[0])
    print (num_stn, num_his, num_fct)

    mos_data = Squential_read_var(output_size, mos_file[0], num_stn, num_his, num_fct)
 


    # .... for one station ( nvar, nstn, nhis, nfct ) --> (nvar, nhis, nfct)
    # .. 37 is 47108
    #mos_data = mos_data[:,stn_idx,:,:]

    print ('print read pmos data shape= ', mos_data.shape)

    reshape_pmos_x = np.swapaxes(mos_data,0,2)
    reshape_pmos_x = np.swapaxes(reshape_pmos_x,0,1)

    #print ('print reshape_pmos_y ' , reshape_pmos_x.shape)
#    print 'print test_x ' , test_x.shape
#    print 'print test_y ' , test_y.shape

    #dropna_pmos_x, dropna_test_x = Remove_nan_batch(reshape_pmos_x, test_x, 
    #                                                    input_size, num_fct, output_size)
    #dropna_pmos_x, dropna_test_y = Remove_nan_batch(reshape_pmos_x, test_y, 
    #                                                    input_size, num_fct, output_size)


#    print 'print dropna_x ' , test_x.shape
#    print 'print dropna_y ' , test_y.shape
#    print 'print dropna_p ' , dropna_pmos_x.shape

    return mos_data
    #return reshape_pmos_x
    #return test_x, test_y, reshape_pmos_x
