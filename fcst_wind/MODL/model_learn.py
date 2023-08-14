import numpy as np
import pandas as pd
import sys

sys.path.insert(0, '../config/')
from global_params import variable_info
#from config.global_params import variable_info

sys.path.insert(0, './INC/')
from data_split import data_split


# 인풋 준비
nwp_file = "../DAIO/nwp_data_47105"
obs_file = "../DAIO/obs_data_47105"
nwp_data = np.load(nwp_file)
obs_data = np.load(obs_file)
print("="*50, "load data shape")
print(nwp_data.shape)
print(obs_data.shape)


# train([21.01,04, 22.01,04]) / test([23.01,04]) 분할  
class_split = data_split(nwp_data, obs_data)
train_nwp, test_nwp, train_obs, test_obs = class_split.get_split_data()
print("="*50, "split data shape")
print(train_nwp.shape)
print(train_obs.shape)
print(test_nwp.shape)
print(test_obs.shape)



# 결측제거
missing_nwp_train = set(np.where(np.isnan(train_nwp))[0])
missing_obs_train = set(np.where(np.isnan(train_obs))[0])
missing_all_train = list(missing_nwp_train | missing_obs_train)
print("결측 합계: ", len(missing_all_train))
dm_nwp_train = np.delete(train_nwp, missing_all_train, 0)
dm_obs_train = np.delete(train_obs, missing_all_train, 0)
print("shape of after drop")
print(dm_nwp_train.shape)
print(dm_obs_train.shape)

missing_nwp_test = set(np.where(np.isnan(test_nwp))[0])
missing_obs_test = set(np.where(np.isnan(test_obs))[0])
missing_all_test = list(missing_nwp_test | missing_obs_test)
print("결측 합계: ", len(missing_all_test))
dm_nwp_test = np.delete(test_nwp, missing_all_test, 0)
dm_obs_test = np.delete(test_obs, missing_all_test, 0)
print("shape of after drop")
print(dm_nwp_test.shape)
print(dm_obs_test.shape)


# 변수선택
sel_var = ['NDNSW_surface', 'UGRD_10m', 'VGRD_10m', 'RH_1_5ma', 'MAXGUST_0m', 'PRMSL_meansealevel']
var_list_dict = list(variable_info.keys())
var_index = [ var_list_dict.index(i) for i in sel_var ]
#print(var_list_dict)
#print(var_index)
sel_dm_nwp_train = dm_nwp_train[:,:,var_index]
sel_dm_nwp_test = dm_nwp_test[:,:,var_index]
print("="*50, "drop data shape")
print(sel_dm_nwp_train.shape)
print(dm_obs_train.shape)
print(sel_dm_nwp_test.shape)
print(dm_obs_test.shape)








# .. 스케일링 및 데이터 분할
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#-------------------------------------------------------------------------
# .. Normalize

output_size = 2

# .. initialaize
tr_b, tr_s, tr_f = sel_dm_nwp_train.shape[0], sel_dm_nwp_train.shape[1], sel_dm_nwp_train.shape[2]      
ts_b, ts_s, ts_f = sel_dm_nwp_test.shape[0], sel_dm_nwp_test.shape[1], sel_dm_nwp_test.shape[2]      

# .. get restorator with obs range
nwp_scaler = MinMaxScaler()   # copy default true
obs_scaler = MinMaxScaler()
nwp_scaler.fit(sel_dm_nwp_train.view().reshape(tr_b*tr_s, tr_f))
obs_scaler.fit(dm_obs_train.view().reshape(tr_b*tr_s, output_size))

# .. feature normalize   ( train seq, feature = test seq, feature )
nor_dm_nwp_train = nwp_scaler.transform(sel_dm_nwp_train.reshape(tr_b*tr_s, tr_f))
nor_dm_nwp_train = nor_dm_nwp_train.reshape(tr_b,tr_s,tr_f)
nor_dm_obs_train = obs_scaler.transform(dm_obs_train.reshape(tr_b*tr_s, output_size))
nor_dm_obs_train = nor_dm_obs_train.reshape(tr_b,tr_s, output_size)

nor_dm_nwp_test = nwp_scaler.transform(sel_dm_nwp_test.reshape(ts_b*ts_s, ts_f))
nor_dm_nwp_test = nor_dm_nwp_test.reshape(ts_b,ts_s,ts_f)
nor_dm_obs_test = obs_scaler.transform(dm_obs_test.reshape(ts_b*ts_s, output_size))
nor_dm_obs_test = nor_dm_obs_test.reshape(ts_b,ts_s, output_size)

nor_dm_nwp_train = nor_dm_nwp_train[:,1::,:]
nor_dm_obs_train = nor_dm_obs_train[:,1::,:]

nor_dm_nwp_test = nor_dm_nwp_test[:,1::,:]
nor_dm_obs_test = nor_dm_obs_test[:,1::,:]

print ('---------- Final training data shape')
print(type(nor_dm_nwp_train))
print ('tran nwp : ', nor_dm_nwp_train.shape)
print ('tran obs : ', nor_dm_obs_train.shape)
print ('test nwp : ', nor_dm_nwp_test.shape)
print ('test obs : ', nor_dm_obs_test.shape)




import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model as plm
from tensorflow.keras.activations import swish
from tensorflow.keras.layers import Dense, TimeDistributed
from tensorflow.keras import Input, Model, callbacks
from tensorflow.keras import metrics

from tcn import TCN, tcn_full_summary


import os
import joblib

#import keras
#from keras.wrappers.scikit_learn import KerasRegressor
#from tensorflow.keras.models import model_from_json
from keras.callbacks import CSVLogger

#-------------------------------------------------------------------------
# .. Set configure

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.compat.v1.Session(config=config))


#-------------------------------------------------------------------------
# .. Data set

element = 'ALLV'
name_list = "./SHEL/namelist.input"

hp_lr = 0.009
hp_pd = 'same'
hp_ns = 1
#hp_dl = [1,2,4,8,16,32,48]
#hp_dl = [1,2,4,8,16]
hp_dl = [1,2,4]
hp_ldl = hp_dl[-1] # last dilation factor to make name of save model
hp_bn = True
hp_nf = 85
hp_dr = 0.07
hp_ks = 3

input_size = 6
output_size = 2
num_fct = 48
batch_size = 8
num_epoch = 1000
dev_stn_id = 47105
tran_data_per = "2101_2104_2201_2204"


exp_name = "CNTL"
csv_outdir = './DAOU/LOSS/' + exp_name + '/'
model_outdir = './DAOU/MODL/' + exp_name + '/'
scalr_outdir = './DAOU/SCAL/' + exp_name + '/'
gifd_outdir = './GIFD/' + exp_name + '/'
log_outdir = './DAOU/LOGF/' + exp_name + '/'
mplot_outdir = './GIFD/' + exp_name + '/'


if os.path.exists(csv_outdir) != True: os.makedirs(csv_outdir)
if os.path.exists(model_outdir) != True: os.makedirs(model_outdir)
if os.path.exists(scalr_outdir) != True: os.makedirs(scalr_outdir)
if os.path.exists(gifd_outdir) != True: os.makedirs(gifd_outdir)
if os.path.exists(log_outdir) != True: os.makedirs(log_outdir)


#-------------------------------------------------------------------------
# .. Set model label

make_option = '_e' + str(num_epoch) + '_bs' + str(batch_size) + '_lr' + str(hp_lr) + \
              '_nf' + str(hp_nf) + '_pd' + hp_pd + '_ks' + str(hp_ks) + '_dr'+str(hp_dr) + \
              '_dl' + str(hp_ldl) + '_ns' + str(hp_ns) 
model_name = 'tcn_modl_' + 'var' + str(input_size) + make_option + \
             '_' + tran_data_per + '_' + str(dev_stn_id) + '.h5'
loss_name = 'loss_' + 'var' + str(input_size) + make_option + \
             '_' + tran_data_per + '_' + str(dev_stn_id)
scalr_name = 'sclr_' + 'var' + str(input_size) + make_option + \
             '_' + tran_data_per + '_' + str(dev_stn_id) + '.pkl'
log_name = 'log_' + 'var' + str(input_size) + make_option + \
             '_' + tran_data_per + '_' + str(dev_stn_id) + '.csv'
modelplot_name = 'mplt_' + 'var' + str(input_size) + make_option + \
             '_' + tran_data_per + '_' + str(dev_stn_id) + '.png'


#-------------------------------------------------------------------------
# .. Set Model

# 
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:

    try:

        for gpu in gpus:

            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:

        print(e)
    

#-------------------------------------------------------------------------
# .. Set custom metrics
def me(y_true, y_pred):
    return K.mean((y_pred - y_true), axis=-1)


# .. Set batch for whole data


# .. create model - API
i = Input( batch_shape=(None, num_fct, input_size) ) 
o = TCN(return_sequences=True, 
        activation=swish, 
        nb_filters=hp_nf, 
        padding=hp_pd,
        use_batch_norm = hp_bn,
        nb_stacks=hp_ns,
        dropout_rate=hp_dr,
        kernel_size=hp_ks,
        use_skip_connections=True,
        dilations=hp_dl
        )(i)
o = TimeDistributed(Dense(output_size, activation='linear'))(o)

adam = optimizers.Adam(learning_rate=hp_lr)
m= Model(inputs=[i], outputs=[o])
#m.compile(optimizer=adam, loss='mse')
m.compile(optimizer=adam, loss='mse', metrics=[metrics.RootMeanSquaredError(name="root_mean_squared_error"),
                                               metrics.CosineSimilarity(name="cosine_similarity")])

#tcn_full_summary(m, expand_residual_blocks=True)
m.summary()
plm(m, to_file=mplot_outdir + modelplot_name, show_shapes=True)



#-------------------------------------------------------------------------
# .. save scaler
joblib.dump(nwp_scaler, scalr_outdir + 'nwp_' + scalr_name)
joblib.dump(obs_scaler, scalr_outdir + 'obs_' + scalr_name)



#-------------------------------------------------------------------------
# .. training model

csv_logger = CSVLogger(log_outdir + log_name, append=True, separator=';')

#callbacks_list = [ callbacks.ModelCheckpoint(filepath=model_outdir + model_name, monitor='val_loss', save_best_only=True, verbose=1) ]
callbacks_list = [ callbacks.ModelCheckpoint(filepath=model_outdir + model_name, monitor='loss', save_best_only=True, verbose=1), csv_logger]


hist = m.fit(nor_dm_nwp_train, nor_dm_obs_train,
      epochs=num_epoch,
      batch_size=batch_size, 
      callbacks=callbacks_list,
      shuffle=True)
      #validation_data=(nor_vald_x, nor_vald_y) )


#-------------------------------------------------------------------------
# .. Save Loss history of best model

# loss_df = pd.DataFrame(list(zip(hist.history['loss'],hist.history['val_loss'])),
#                        columns=["tran","eval"])
loss_df = pd.DataFrame(list(hist.history['loss']),
                       columns=["tran"])
loss_df.to_csv(csv_outdir+loss_name)



#-------------------------------------------------------------------------
# .. Clear model

K.clear_session()