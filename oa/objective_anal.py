import numpy as np
import pandas as pd

import numpy as np
import netCDF4 as nc
import glob

import matplotlib.pyplot as plt
from fastbarnes import interpolationS2, interpolation


class fastbarnes_run_1step():

    def __init__(self, nwp_grid, obs_stnd, var_name, run_time):

        self.nwp_grid = nwp_grid
        self.obs_stnd = obs_stnd
        self.var_name = var_name
        self.run_time = run_time

        print( )
        
        try:
            if var_name == "TMP_1_5maboveground":
                nwp_grd = np.array(nc.Dataset(nwp_grid).variables['TMP_1_5maboveground'])[run_time,:,:] - 273.15
            else:
                nwp_grd = np.array(nc.Dataset(nwp_grid).variables['TMP_1_5maboveground'])[run_time,:,:]

            print("Read nwp: ", nwp_grid)
            print("nc file info: ", nwp_grd)
            print("var Min/Max: ", nwp_grd.min(), nwp_grd.max())

            nwp_lat = np.load("../fcst_wind/DAIO/nwp/ldaps_lat_grid")
            nwp_lon = np.load("../fcst_wind/DAIO/nwp/ldaps_lon_grid")

            print("lat shape: ", nwp_lat.shape)
            print("lon shape: ", nwp_lon.shape)
            print(nwp_lat[0,0], nwp_lat[-1,-1])
            print(nwp_lon[0,0], nwp_lon[-1,-1])


        except Exception as e:
            print(e)


# obs stn for 불규칙 데이터
stn_latlon = {"47105": [37.7515, 128.891]}
obs_lat = stn_latlon['47105'][0]
obs_lon = stn_latlon['47105'][1]
obs_stnd = np.load("../fcst_wind/DAIO/obs_data_47105")
print(obs_stnd.shape)


# select U & V
#if sel_v == 'u':
#    nwp_grid[:,:,0]


# # 37~39, 127~129
print(nwp_lat.shape)
print(nwp_lon.shape)

lat_mask = (nwp_lat >= 37.0) & (nwp_lat <= 39.0)
lon_mask = (nwp_lon >= 127.0) & (nwp_lon <= 129.0)
ll_mask = lat_mask & lon_mask
#mask = ( (a<1) & (b<2) )
print(lat_mask)
print(lon_mask)
print(ll_mask)

ll_mask_idx = np.where(ll_mask==True)
print(ll_mask_idx[0])
print(ll_mask_idx[1])
y_idx = list(set(ll_mask_idx[0]))
x_idx = list(set(ll_mask_idx[1]))
sub_nwp_lat = nwp_lat[y_idx[0]:y_idx[-1]+1,x_idx[0]:x_idx[-1]+1]
sub_nwp_lon = nwp_lon[y_idx[0]:y_idx[-1]+1,x_idx[0]:x_idx[-1]+1]
sub_nwp_grd_t = nwp_grd_t[y_idx[0]:y_idx[-1]+1,x_idx[0]:x_idx[-1]+1]
print(sub_nwp_lat.shape)
print(sub_nwp_lon.shape)
print(sub_nwp_grd_t.shape)

print(sub_nwp_grd_t.min(), sub_nwp_grd_t.max())
print(sub_nwp_lat.min(), sub_nwp_lat.max())
print(sub_nwp_lon.min(), sub_nwp_lon.max())

# # .. test
# a = np.array([[2,2,2,2],
#               [2,0,1,2],
#               [2,1,0,2],
#               [2,2,2,2]])

# b = np.array([[2,2,2,2],
#               [2,1,0,2],
#               [2,0,1,2],
#               [2,2,2,2]])

# a_mask = (a < 2) & (a >=0)
# b_mask = (b < 2) & (b >=0)
# c_mask = a_mask & b_mask
# #mask = ( (a<1) & (b<2) )
# print(a_mask)
# print(b_mask)
# print(c_mask)
# print(type(c_mask))
# c_mask_idx = np.where(c_mask==True)
# print(c_mask_idx[0])
# print(c_mask_idx[1])
# y_idx = list(set(c_mask_idx[0]))
# x_idx = list(set(c_mask_idx[1]))
# c = a[y_idx[0]:y_idx[-1]+1,x_idx[0]:x_idx[-1]+1]
# print(c.shape)
# print(c)

# 인풋 shape 만들기
flatten_input = []
for i in range(sub_nwp_lat.shape[0]): # lat
    for j in range(sub_nwp_lat.shape[1]): # lon
        flatten_input.append([ sub_nwp_lon[i,j], sub_nwp_lat[i,j], sub_nwp_grd_t[i,j] ])
flatten_input.append([obs_lon, obs_lat, obs_stnd[10,0,0]])


# asarray(copy=False)
print("ldps points: ",len(flatten_input))
input_data = np.asarray(flatten_input)
print(input_data[0:5])

# extract attribute 
lon_lat_data = input_data[:, 0:2]
qff_values = input_data[:, 2]

print(lon_lat_data.shape)
print(qff_values.shape)
print(np.min(lon_lat_data[:,0]), ',', np.min(lon_lat_data[:,1]))
print(np.max(lon_lat_data[:,0]), ',', np.max(lon_lat_data[:,1]))
#print(qff_values[0])
print(np.min(qff_values), np.max(qff_values))


plt.figure(figsize=(5, 5))

gridX = np.arange(127, 129, 1)
gridY = np.arange(37, 39, 1)
levels = np.arange(0, 20, 0.1)
#cs = plt.contour(lon_lat_data[:][1], lon_lat_data[:][0], qff_values, levels)
#plt.clabel(cs, levels[::2], fmt='%d', fontsize=9)

plt.scatter(lon_lat_data[:, 0], lon_lat_data[:, 1], color='red', s=1, marker='.')

plt.show()




lon_dist = round(abs(lon_lat_data[:,0].min() - lon_lat_data[:,0].max()))
lat_dist = round(abs(lon_lat_data[:,1].min() - lon_lat_data[:,1].max()))
step = 0.005
x0 = np.asarray([round(lon_lat_data[0,0]), round(lon_lat_data[0,1])], dtype=np.float64)
size = (int(lon_dist / step), int(lat_dist / step))
x_e_lon = x0[0] + (step*size[0])
x_e_lat = x0[1] + (step*size[1])
print("start: ", x0)
print("lon dist: ", lon_dist)
print("lat dist: ", lat_dist)
print("step: ", step)
print("oa dim: ", size)
print("end: ", x_e_lon, x_e_lat)



# calculate Barnes interpolation



sigma = 0.01
#field = interpolation.barnes(lon_lat_data, qff_values, sigma, x0, step, size)
field = interpolation.barnes(lon_lat_data, qff_values, sigma, x0, step, size)
print(type(field))
print(np.round(field, 2))
print(np.min(field), np.max(field))
print(field.shape)

#field.shape
#field_1 = np.where( (field<=20.) & (field>=-20.) , field, 0.)


# draw graphic with labeled contours and scattered sample points

plt.figure(figsize=(5, 5))

gridX = np.arange(x0[0], x0[0]+size[1]*step, step)
gridY = np.arange(x0[1], x0[1]+size[0]*step, step)
#print(gridX, gridY)
levels = np.arange(0, 20, 0.1)
cs = plt.contour(gridX, gridY, field, levels)
plt.clabel(cs, levels, fmt='%d', fontsize=9)

plt.scatter(lon_lat_data[:, 0], lon_lat_data[:, 1], color='red', s=1, marker='.')

plt.show()