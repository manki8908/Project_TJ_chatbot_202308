import pandas as pd
import numpy as np

class data_split():


    def __init__(self, nwp, obs):
        # .. 날짜 초기화
        self.all_set_dates = pd.date_range('2021-01-14 09:00:00', '2023-05-31 09:00:00', freq='D')
        self.all_set_dates = self.all_set_dates.strftime("%Y%m%d00")

        self.train_202101_set_dates = pd.date_range('2021-01-14 09:00:00', '2021-01-31 09:00:00', freq='D')
        self.train_202101_set_dates = self.train_202101_set_dates.strftime("%Y%m%d00")
        self.train_202104_set_dates = pd.date_range('2021-04-01 09:00:00', '2021-04-30 09:00:00', freq='D')
        self.train_202104_set_dates = self.train_202104_set_dates.strftime("%Y%m%d00")
        self.train_202201_set_dates = pd.date_range('2022-01-01 09:00:00', '2022-01-31 09:00:00', freq='D')
        self.train_202201_set_dates = self.train_202201_set_dates.strftime("%Y%m%d00")
        self.train_202204_set_dates = pd.date_range('2022-04-01 09:00:00', '2022-04-30 09:00:00', freq='D')
        self.train_202204_set_dates = self.train_202204_set_dates.strftime("%Y%m%d00")

        self.test_202301_set_dates = pd.date_range('2023-01-01 09:00:00', '2023-01-31 09:00:00', freq='D')
        self.test_202301_set_dates = self.test_202301_set_dates.strftime("%Y%m%d00")
        self.test_202304_set_dates = pd.date_range('2023-04-01 09:00:00', '2023-04-30 09:00:00', freq='D')
        self.test_202304_set_dates = self.test_202304_set_dates.strftime("%Y%m%d00")

        self.nwp = nwp
        self.obs = obs


    def get_time_index(self, find_period, all_period):
  
        test = [ all_period.get_loc(date) for date in find_period ]
        return test


    def get_split_data(self,):
        train_idx, test_idx = [], []

        dummy = self.get_time_index(self.train_202101_set_dates, self.all_set_dates)
        train_idx.extend(dummy)
        dummy = self.get_time_index(self.train_202104_set_dates, self.all_set_dates)
        train_idx.extend(dummy)
        dummy = self.get_time_index(self.train_202201_set_dates, self.all_set_dates)
        train_idx.extend(dummy)
        dummy = self.get_time_index(self.train_202204_set_dates, self.all_set_dates)
        train_idx.extend(dummy)

        dummy = self.get_time_index(self.test_202301_set_dates, self.all_set_dates)
        test_idx.extend(dummy)
        dummy = self.get_time_index(self.test_202304_set_dates, self.all_set_dates)
        test_idx.extend(dummy)

        train_nwp = self.nwp[train_idx, :, :]
        train_obs = self.obs[train_idx, :, :]
        test_nwp = self.nwp[test_idx, :, :]
        test_obs = self.obs[test_idx, :, :]

        return train_nwp, test_nwp, train_obs, test_obs

        

