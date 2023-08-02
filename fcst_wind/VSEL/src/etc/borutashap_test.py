import numpy as np
import pandas as pd
from BorutaShap import BorutaShap
#from xgboost import XGBRegressor 
from sklearn.ensemble import RandomForestRegressor


## Load data
#nwp = np.load("../dat/test_nwp_2016050100-2021043000-24-1605-2104.npz")['value']
#obs = np.load("../dat/test_obs_2016050100-2021043000-24-1605-2104.npz")['value']
#header = np.load("../dat/test_nwp_2016050100-2021043000-24-1605-2104.npz")['value_name']
nwp = np.load("../dat/test_nwp_sample.npz")['value']
obs = np.load("../dat/test_obs_sample.npz")['value']
header = np.load("../dat/test_nwp_sample.npz")['value_name']
header = header.tolist()

print("nwp shape: ", nwp.shape)
print("obs shape: ", obs.shape)
print(header)

pd_input = pd.DataFrame(data=nwp, columns=header)
pd_target = np.array(obs[:,0])

print(pd_input)
print(pd_target)



#rf = RandomForestRegressor(n_jobs=-1, criterion='mse', max_depth=30,  ## max_depth=5
#                           min_samples_split=3, max_features=0.33, random_state=1)
rf = RandomForestRegressor(n_jobs=-1, criterion='mse', random_state=1)
#xgb = XGBRegressor() 

feature_selector = BorutaShap(model = rf, importance_measure='shap', classification=False)
feature_selector.fit(X=pd_input, y=pd_target, n_trials=100 )
feature_selector.plot(which_features='all')
