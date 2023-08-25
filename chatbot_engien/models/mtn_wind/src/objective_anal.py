import numpy as np
import pandas as pd

import numpy as np
import netCDF4 as nc
import glob

import matplotlib.pyplot as plt
from fastbarnes import interpolationS2, interpolation


class fastbarnes_run_1ele():

    def __init__(self, nwp_name, nwp_var, obs_name, obs_dict, obs_var, run_time):

        self.nwp_name = nwp_name
        self.nwp_var = nwp_var
        self.obs_name = obs_name
        self.obs_dic = list(obs_dict.items()) #{"47105": [37.7515, 128.891]}
        self.obs_var = obs_var
        self.run_time = run_time
        
        self.obs_nid = self.obs_dic[0][0]
        self.obs_lat = self.obs_dic[0][1][0]
        self.obs_lon = self.obs_dic[0][1][1]

        print("nwp file= ", self.nwp_name )
        print("obs_name= ", self.obs_name )
        print("nwp_var= ", self.nwp_var )
        print("obs_info= ", self.obs_dic )
        print("obs_var= ", self.obs_var )
        print("run_time= ", self.run_time )

        self.nwp_lat = None
        self.nwp_lon = None
        self.bcr_obs = None
        self.nwp_grd = None
      

    def read_input(self,):
        
        try:

            print("Read nwp file: ", self.nwp_name)            
            # .. nwp read
            if self.nwp_var == "TMP_1_5maboveground":
                self.nwp_grd = np.array(nc.Dataset(self.nwp_name).variables[self.nwp_var])[self.run_time,:,:] - 273.15
            else:
                self.nwp_grd = np.array(nc.Dataset(self.nwp_name).variables[self.nwp_var])[self.run_time,:,:]

            #print("nc file info: ", self.nwp_grd)
            print("var Min/Max: ", self.nwp_grd.min(), self.nwp_grd.max())
            

            self.nwp_lat = np.array(nc.Dataset(self.nwp_name).variables['latitude'])
            self.nwp_lon = np.array(nc.Dataset(self.nwp_name).variables['longitude'])

            print("lat shape: ", self.nwp_lat.shape)
            print("lon shape: ", self.nwp_lon.shape)
            print(self.nwp_lat[0,0], self.nwp_lat[-1,-1])
            print(self.nwp_lon[0,0], self.nwp_lon[-1,-1])

            print("--- nwp read complete")

            # .. obs read
            # use real obs for test
            #self.bcr_obs = np.load("../fcst_wind/DAIO/obs_data_47105")[10,self.run_time,:] # u,v
            self.bcr_obs = np.load(self.obs_name)[10,self.run_time,self.obs_var] # u,v
            print("obs shape: ", self.bcr_obs.shape)
            print(self.bcr_obs)

        except Exception as e:
            print(e)


    def extract_subarea(self, slat, elat, slon, elon):

        # set sub area mask
        lat_mask = (self.nwp_lat >= slat) & (self.nwp_lat <= elat)
        lon_mask = (self.nwp_lon >= slon) & (self.nwp_lon <= elon)
        ll_mask = lat_mask & lon_mask
        ll_mask_idx = np.where(ll_mask==True)
        y_idx = list(set(ll_mask_idx[0]))
        x_idx = list(set(ll_mask_idx[1]))
        #print(lat_mask)
        #print(lon_mask)
        #print(ll_mask)     
        #print(ll_mask_idx[0])
        #print(ll_mask_idx[1])
        
        # extract
        self.sub_nwp_lat = self.nwp_lat[y_idx[0]:y_idx[-1]+1,x_idx[0]:x_idx[-1]+1]
        self.sub_nwp_lon = self.nwp_lon[y_idx[0]:y_idx[-1]+1,x_idx[0]:x_idx[-1]+1]
        self.sub_nwp_grd = self.nwp_grd[y_idx[0]:y_idx[-1]+1,x_idx[0]:x_idx[-1]+1]
        print("sub lat: ", self.sub_nwp_lat.shape)
        print("sub lon: ", self.sub_nwp_lon.shape)
        print("sub nwp: ", self.sub_nwp_grd.shape)

        print(self.sub_nwp_lat.min(), self.sub_nwp_lat.max())
        print(self.sub_nwp_lon.min(), self.sub_nwp_lon.max())
        print(self.sub_nwp_grd.min(), self.sub_nwp_grd.max())


    def barnes_run(self,):
        
        # .. make shape
        flatten_input = []
        for i in range(self.sub_nwp_lat.shape[0]): # lat
            for j in range(self.sub_nwp_lat.shape[1]): # lon
                flatten_input.append([ self.sub_nwp_lon[i,j], self.sub_nwp_lat[i,j], self.sub_nwp_grd[i,j] ])
        flatten_input.append([self.obs_lon, self.obs_lat, self.bcr_obs])
        print("# analysis points: ",len(flatten_input))


        # .. extract attribute  
        input_data = np.asarray(flatten_input) # asarray(copy=False)
        lon_lat_data = input_data[:, 0:2]
        qff_values = input_data[:, 2]

        print("lat, lon data: ", lon_lat_data.shape)
        print("value: ", qff_values.shape)
        print(np.min(lon_lat_data[:,0]), ',', np.min(lon_lat_data[:,1]))
        print(np.max(lon_lat_data[:,0]), ',', np.max(lon_lat_data[:,1]))
        print(np.min(qff_values), np.max(qff_values))

        # .. check input domain
        #plt.figure(figsize=(5, 5))
        #plt.scatter(lon_lat_data[:, 0], lon_lat_data[:, 1], color='red', s=1, marker='.')
        #plt.show()

        # .. set barnes 
        lon_dist = round(abs(lon_lat_data[:,0].min() - lon_lat_data[:,0].max()))
        lat_dist = round(abs(lon_lat_data[:,1].min() - lon_lat_data[:,1].max()))
        step = 0.005
        x0 = np.asarray([round(lon_lat_data[0,0]), round(lon_lat_data[0,1])], dtype=np.float64)
        size = (int(lon_dist / step), int(lat_dist / step))
        x_e_lon = x0[0] + (step*size[0])
        x_e_lat = x0[1] + (step*size[1])

        print("=== barnes setting")
        print("start: ", x0)
        print("lon dist: ", lon_dist)
        print("lat dist: ", lat_dist)
        print("step: ", step)
        print("oa dim: ", size)
        print("end: ", x_e_lon, x_e_lat)

        # .. run Barnes interpolation
        sigma = 0.005
        field = interpolation.barnes(lon_lat_data, qff_values, sigma, x0, step, size)
        #print(type(field))
        #print(np.round(field, 2))

        

        # .. draw graphic with labeled contours and scattered sample points
        #
        #plt.figure(figsize=(5, 5))
        #
        gridX = np.arange(x0[0], x0[0]+size[0]*step, step)
        gridY = np.arange(x0[1], x0[1]+size[1]*step, step)
        ##print(gridX, gridY)
        #levels = np.arange(0, 20, 0.1)
        #cs = plt.contour(gridX, gridY, field, levels)
        #plt.clabel(cs, levels, fmt='%d', fontsize=9)
        #
        #plt.scatter(lon_lat_data[:, 0], lon_lat_data[:, 1], color='red', s=1, marker='.')
        #
        #plt.show()

        print("=== field out")
        print("field shape: ", field.shape)
        print("gridX shape: ", gridX.shape)
        print("gridY shape: ", gridY.shape)
        print(np.min(field), np.max(field))

        return field, gridX, gridY