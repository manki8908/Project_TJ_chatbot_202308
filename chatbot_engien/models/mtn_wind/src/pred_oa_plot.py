import Ngl,Nio
import numpy as np
import pandas as pd
import sys
#sys.path.insert(0, '/Users/mankikim/JOB/prj_mtn2/fcst_wind/MODL/INC/')
from data_load import data_load


anal_time='2023041000' # UTC


data_name_tsnwp = '/Users/mankikim/JOB/prj_mtn2/fcst_wind/DAIO/nwp_data_47105'
data_name_tsobs = '/Users/mankikim/JOB/prj_mtn2/fcst_wind/DAIO/obs_data_47105'
data_name_grnwp = "/Users/mankikim/Desktop/DATA_LINK/l015_v070_erlo_unis_20kind.2023041000.nc"

timestep = 13

# 시계열 데이터
_, sel_dm_nwp_test, _, dm_obs_test = data_load(data_name_tsnwp, data_name_tsobs)
tsnwp = sel_dm_nwp_test[timestep,:,:]
tsobs = dm_obs_test[timestep,:,:]
print(tsnwp.shape)
print(tsobs.shape)

#--  open file
f = Nio.open_file(data_name_grnwp, "r")
u = f.variables["UGRD_10maboveground"][:,:,:]
v = f.variables["VGRD_10maboveground"][:,:,:]
lat = f.variables["latitude"]
lon = f.variables["longitude"]
print(u.shape)
print(v.shape)


import joblib

# load scaler
nwp_scaler = joblib.load('nwp_sclr_var6_e1000_bs8_lr0.009_nf85_pdsame_ks3_dr0.07_dl4_ns1_2101_2104_2201_2204_47105.pkl')
obs_scaler = joblib.load('obs_sclr_var6_e1000_bs8_lr0.009_nf85_pdsame_ks3_dr0.07_dl4_ns1_2101_2104_2201_2204_47105.pkl')


# .. load data
s, f = tsnwp.shape
print ( "load data shape before prediction: ", s, f )
# .. normalize
nor_test_x = nwp_scaler.transform(tsnwp.reshape(1*s,f))
nor_test_x = nor_test_x.reshape(1,s,f)




#-------------------------------------------------------------------------
# .. Model load

from tensorflow.keras.models import load_model
from tcn import TCN

#model_name = "../MODL/DAOU/MODL/CNTL/tcn_modl_var6_e1000_bs8_lr0.009_nf87_pdsame_ks6_dr0.07_dl48_ns1_2101_2104_2201_2204_47105.h5"

#hyper band
model_name = "./tcn_modl_var6_e1000_bs8_lr0.009_nf85_pdsame_ks3_dr0.07_dl4_ns1_2101_2104_2201_2204_47105.h5"

#bayesian
#model_name = "../MODL/DAOU/MODL/CNTL/tcn_modl_var6_e1000_bs8_lr0.009_nf95_pdsame_ks3_dr0.07_dl4_ns1_2101_2104_2201_2204_47105.h5"

print ("load_model: ", model_name)
model = load_model(model_name, custom_objects={'TCN':TCN} )


nor_pred_test_y = model.predict(nor_test_x)
inv_pred_test = obs_scaler.inverse_transform(nor_pred_test_y.reshape(1*s, 2))
inv_pred_test = inv_pred_test.reshape(1,s, 2)
print(inv_pred_test.shape)




# nwp 오른쪽 끝 열에 _fillvalue 존재 --> sub area만 계산
from objective_anal import fastbarnes_run_1ele

#nwp_file = r"D:\KMK_DATA\NWP\l015_v070_erlo_unis_20kind.2023041000.nc"
nwp_file = "/Users/mankikim/Desktop/DATA_LINK/l015_v070_erlo_unis_20kind.2023041000.nc"
var_list = {"UGRD_10maboveground": 0, "VGRD_10maboveground": 1}
stn_info = {"47105": [37.7515, 128.891]}

uv_OA_field = u_OA_field = np.ndarray(shape=(400,400,2), dtype=np.float_)
for i, (key, value) in enumerate(var_list.items()):
    fb_run = fastbarnes_run_1ele(nwp_name=nwp_file, 
                                nwp_var=key, 
                                obs_name="/Users/mankikim/JOB/prj_mtn2/fcst_wind/DAIO/obs_data_47105", 
                                obs_dict=stn_info, 
                                obs_var=value,  # 0: u, 1:v
                                run_time=timestep) # 0: 0900 KST
    # .. input read
    fb_run.read_input()
    # .. sub area
    fb_run.extract_subarea(slat=37., elat=39., slon=127., elon=129.)
    # .. return O.A value
    field, gridX, gridY = fb_run.barnes_run()
    u_OA_field[:,:,i] = field

# write 하면서 좌우 반전이 생긴것 같음 i,j 살펴보기
#u_OA_field = u_OA_field.swapaxes(0,1)
f = open('./uv_OA_field_test'+str(timestep), 'w')
for j in range(len(gridY)):
    for i in range(len(gridX)):
        print(j,i,gridY[j],gridX[i],u_OA_field[j,i,0], u_OA_field[j,i,1], sep=',', file=f)
        #print(j,i,gridY[j],gridX[i],u_OA_field[i,j,0], u_OA_field[i,j,1], sep=',', file=f)
f.close()

np.savez( "./uv_OA_field_test_"+str(timestep), value=uv_OA_field, stn_info=stn_info, gridX=gridX, gridY=gridY)