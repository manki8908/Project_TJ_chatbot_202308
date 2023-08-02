import numpy as np
#from BorutaShap import BorutaShap
import xgboost
#from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import os
import sys
import time
from sklearn.multioutput import MultiOutputRegressor

#.. local
sys.path.insert(0, './')
from plot_importance import plot_importance_barh, plot_importance_hline

## Load data
#nwp = np.load("../dat/test_nwp_2016050100-2021043000-24-1605-2104.npz")['value']
#obs = np.load("../dat/test_obs_2016050100-2021043000-24-1605-2104.npz")['value']
#header = np.load("../dat/test_nwp_2016050100-2021043000-24-1605-2104.npz")['value_name']
nwp = np.load("../dat/test_nwp_sample.npz")['value']
obs = np.load("../dat/test_obs_sample.npz")['value']
header = np.load("../dat/test_nwp_sample.npz")['value_name']
header = header.tolist()


print (np.shape(nwp), np.shape(obs), header, type(header)) # variable, day, forecast_time
X = nwp
y = obs

print (X, y, np.shape(X), np.shape(y)) # forecast_time * day, variable


#forest = RandomForestRegressor(n_jobs=-1, criterion='mse', max_depth=30,  ## max_depth=5
#                           min_samples_split=3, max_features=0.33, random_state=0)

#forest = RandomForestRegressor(n_estimators= 10, n_jobs=-1, random_state=0)
print("XGBoost Regressor ing..")
forest = MultiOutputRegressor(xgboost.XGBRegressor(n_estimators= 10, n_jobs=-1, random_state=0))
forest.fit(X, y)


print( "Features sorted by their score:")
importances_u = forest.estimators_[0].feature_importances_
indices_u = np.argsort(importances_u)[::-1]
importances_v = forest.estimators_[1].feature_importances_
indices_v = np.argsort(importances_v)[::-1]


sorted_importances_u= []
sorted_feature_name_u= []
sorted_importances_v= []
sorted_feature_name_v= []
# Print the feature ranking
print("Feature ranking for U-component:")
feature_name = header
for f in range(X.shape[1]-1,-1,-1):
    print("%d. feature %d %s (%f)" % (f + 1, indices_u[f], feature_name[indices_u[f]], importances_u[indices_u[f]]) )
    sorted_importances_u.append(importances_u[indices_u[f]])
    sorted_feature_name_u.append(feature_name[indices_u[f]])
for f in range(X.shape[1]-1,-1,-1):
    print("%d. feature %d %s (%f)" % (f + 1, indices_v[f], feature_name[indices_v[f]], importances_v[indices_v[f]]) )
    sorted_importances_v.append(importances_v[indices_v[f]])
    sorted_feature_name_v.append(feature_name[indices_v[f]])
    

save_name = "../out/xgb_uuu_barh"
plot_importance_barh(sorted_feature_name_u, sorted_importances_u, save_name)
save_name = "../out/xgb_uuu_hline"
plot_importance_hline(sorted_feature_name_u, sorted_importances_u, save_name)
save_name = "../out/xgb_vvv_barh"
plot_importance_barh(sorted_feature_name_v, sorted_importances_v, save_name)
save_name = "../out/xgb_vvv_hline"
plot_importance_hline(sorted_feature_name_v, sorted_importances_v, save_name)
