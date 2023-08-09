#.. module
import os
import numpy as np
import pandas as pd


def func_model_list(load_dir):
        model_list = []
        list = np.array(os.listdir(load_dir))
        find_ep = '1000'
        find_dl = 'dl136'
        find_lr = 'lr0.009'
        find_ns = 'ns1'
        find_ks = 'ks6'
        find_pd = 'pdsame'
        find_nf = 'nf70'

        for i in range(len(list)):
               if list[i].find(find_ep)>0 and list[i].find('h5')>0 and list[i].find(find_pd)>0 and list[i].find(find_ks)>0 and list[i].find(find_ns)>0 and list[i].find(find_dl)>0 and list[i].find(find_lr)>0 and list[i].find(find_nf)>0 :
                  model_list.append(list[i])

        #model_list.sort(key=sort_use_stnid)

        return model_list


def find_intersection_idx(daba_3hr, modl_dir_1hr, return_set):


    # .. read 3hr list
    exists = os.path.isfile(daba_3hr)
    if exists:
       dev_stn_id  = pd.read_fwf(daba_3hr, delimiter=' ', header=2, usecols=[0])
       dev_stn_id  = np.array(dev_stn_id,  dtype=np.int)
    else:
       sys.exit("STOP Error: Could not found : "+ dev_stn_list)
    list_3hr = dev_stn_id[:,0]    

    # .. read 1hr list
    model_list = func_model_list(modl_dir_1hr)
    list_1hr = np.ndarray( shape=(len(model_list)), dtype=np.int )
    list_1hr = np.genfromtxt(model_list, dtype=np.int, delimiter="_", usecols=12)


    dev_stn_3hr = list( list_3hr )
    dev_stn_1hr = list( list_1hr )

   # print("----------------------- 3hr daba print --------------------")
   # print("3hr daba list", dev_stn_3hr)
   # print("# of 3hr daba list", len(dev_stn_3hr))
   # print("----------------------- 1hr modl print --------------------")
   # print("1hr model list", dev_stn_1hr)
   # print("# of 1hr model list", len(dev_stn_1hr))
    
    set_3hr_1hr = list( set(dev_stn_3hr) & set(dev_stn_1hr) )
    
   # print("----------------------- Set_3hr_1hr print --------------------")
   # print(len(set_3hr_1hr))
   # print(set_3hr_1hr)
    
    index_3hr = []
    index_1hr = []
    for i in range(len(set_3hr_1hr)):
        index_3hr.append( dev_stn_3hr.index(set_3hr_1hr[i]) )
        index_1hr.append( dev_stn_1hr.index(set_3hr_1hr[i]) )
    
   # print("----------------------- Index_3hr print --------------------")
   # print(index_3hr)
   # print("----------------------- Index_1hr print --------------------")
   # print(index_1hr)

    if return_set == "3hr": 
       return_index = index_3hr
    else:
       return_index = index_1hr

    return return_index
