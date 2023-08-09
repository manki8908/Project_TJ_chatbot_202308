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



def chek_stn_idx(stn_list, stn_id):

    import sys
    import numpy as np

    is_it_there = 0

    for i in range(len(stn_list)):

        if ( stn_list[i] == stn_id ):
           stn_idx = i
           is_it_there = 1

    if ( is_it_there == 0 ):
       print ("Could not found idx : " + str(stn_id))


def find_stn_id_from_mlist(model_list, stn_id):

    import numpy as np
    import os
    import sys

    is_it_there = 0
    run_idx = -999

    for i in range(len(model_list)):
        if model_list[i].find('_'+str(stn_id)+'.') > 0:
           run_idx = i
           is_it_there = 1
           print("Run stn: ", stn_id, " Idx: ", run_idx, " Find list: ", model_list[i])
        
    if ( is_it_there == 0 ):
       print("STOP Error: Could not found Run stn: " , str(stn_id), " Idx: ", run_idx)

    return run_idx
