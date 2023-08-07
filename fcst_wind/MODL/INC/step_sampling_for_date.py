import numpy as np

def erange(start, end, step):
    a = list(range(start, end, step))
    if a[-1] != end: a.append(end)
    return a


def step_sampling_for_date(re_nwp_stn, re_obs_stn, tran_rate, nbin, seed_fix):


    # .. make clust
    h_dim = re_nwp_stn.shape[0]
    a = erange(0, h_dim, nbin)
    
    
    # .. make h-idx order
    h_idx_list = []
    for i in range(len(a)-1):
        if i != len(a)-2:  h_idx = [i for i in range(a[i],a[i+1],1) ]
        if i == len(a)-2:  h_idx = [i for i in range(a[i],a[i+1],1) ]
        #print(a[i], a[i+1], h_idx)
        h_idx_list.append(h_idx)
    
    
    if len(h_idx_list[-1]) < 5:
       h_idx_list[-2] = h_idx_list[-2] + h_idx_list[-1]
       del(h_idx_list[-1])
    
    #for i in range(len(h_idx_list)):
    #    print(h_idx_list[i])




    # .. split data
    tran_mask_list = []
    vald_mask_list = []
    for i in range(len(h_idx_list)):
        #print(h_idx_list[i][0], h_idx_list[i][-1])
        clust_tran_size = round(len(h_idx_list[i])*tran_rate)
        if seed_fix == True: np.random.seed(0)
        tran_mask = list(np.random.choice(h_idx_list[i], clust_tran_size, replace=False))
        vald_mask = list(set(h_idx_list[i]).difference(tran_mask))
    
        #if i == len(h_idx_list)-1: print(h_idx_list[i])
    
        tran_mask_list = tran_mask_list + tran_mask
        vald_mask_list = vald_mask_list + vald_mask
   
    print("in split, re_nwp_stn shape: ", re_nwp_stn.shape)
 
    tran_nwp_data = re_nwp_stn[tran_mask_list,:,:]
    vald_nwp_data = re_nwp_stn[vald_mask_list,:,:]
    tran_obs_data = re_obs_stn[tran_mask_list,:,:]
    vald_obs_data = re_obs_stn[vald_mask_list,:,:]


    print( "tran nwp shape: ", tran_nwp_data.shape, "tran obs shape: ", tran_obs_data.shape )
    print( "vald nwp shape: ", vald_nwp_data.shape, "vald obs shape: ", vald_obs_data.shape )

    return tran_nwp_data, tran_obs_data, vald_nwp_data, vald_obs_data

