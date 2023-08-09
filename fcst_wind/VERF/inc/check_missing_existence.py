def check_missing_existence(x,y):

    import numpy as np

    #print('------check missing existence-----')
    #print('in x shape: ', x.shape)
    #print('in y shape: ', y.shape)

    #missing_idx_x = np.unique( np.argwhere( x == -999. )[:,0] )
    #missing_idx_y = np.unique( np.argwhere( y == -999. )[:,0] )
    missing_idx_x = np.unique( np.argwhere( np.isnan(x) )[:,0] )
    missing_idx_y = np.unique( np.argwhere( np.isnan(y) )[:,0] )
    missing_idx = sorted( set(missing_idx_x) | set(missing_idx_y) )

    x_count = len(missing_idx_x)
    y_count = len(missing_idx_y)
    missing_count = len(missing_idx)
   
    #print(missing_idx_x)
    #print(missing_idx_y)
    #print ( missing_idx)
    print ("missing count = ", len(missing_idx))

    is_there_missing = False

    if missing_count > 0 :
       is_there_missing = True

    return x_count
