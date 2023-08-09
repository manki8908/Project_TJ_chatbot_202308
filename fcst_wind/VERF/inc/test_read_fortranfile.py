def Read_tran_fortranfile(input_file, NV, NH, NF):

    import numpy as np
    import pandas as pd
    import sys
    from scipy.io import FortranFile

    try:
        print("Read input: ", input_file)
        f = FortranFile(input_file, 'r')
    except:
        sys.exit( "STOP Error: Can not found : ", input_file )

    sdate, edate, idate = f.read_record( np.dtype((np.int32,(4))),
                                   np.dtype((np.int32,(4))),
                                   np.dtype((np.int32)) )
    NV1, NS1, NH1, NF1 = f.read_record( np.dtype((np.int32)),
                                   np.dtype((np.int32)),
                                   np.dtype((np.int32)),
                                   np.dtype((np.int32)) )
    NV1, NS1, NH1, NF1 = np.asscalar(NV1), np.asscalar(NS1), np.asscalar(NH1), np.asscalar(NF1)


    print( "FILE date: ", sdate, edate, idate )
    print( "READ  DIMENSION: NV = ", NV1 )
    print( "READ  DIMENSION: NS = ", NS1 )
    print( "READ  DIMENSION: NH = ", NH1 )
    print( "READ  DIMENSION: NF = ", NF1 )

    print( "USER Request dimension: ")
    print( "USER  DIMENSION: NV = ", NV )
    print( "USER  DIMENSION: NS = ", 1 )
    print( "USER  DIMENSION: NH = ", NH )
    print( "USER  DIMENSION: NF = ", NF )

    if (NH != NH1 or NF != NF1 or NV1 != NV ):
        sys.exit( "Invaild NV,NH,NF in READ_TRN_FOR_LSTM" )

    stn_id = f.read_record( np.dtype((np.int32,(NS1))) )
    ndate = f.read_record( np.dtype((np.int32),(4,NH1)) ).T
    buf = f.read_record( np.dtype((np.float32,(NF1,NH1,NS1,NV1))) ).T



    f.close()

    return buf, stn_id

