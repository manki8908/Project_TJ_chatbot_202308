def find_stn_idx(stn_list, stn_id):

    import sys
    import numpy as np

    is_it_there = 0

    for i in range(len(stn_list)):

        if ( stn_list[i] == stn_id ):
           stn_idx = i 
           is_it_there = 1

    if ( is_it_there == 0 ):
       sys.exit("STOP Error: Could not found idx : " + str(stn_id))

    return stn_idx
