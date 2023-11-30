from snr_noise import noise_pressoce
import  numpy as np
import argparse
import os
import tensorflow as tf

from sklearn.preprocessing import StandardScaler

from utils import MinMax,butter_bandpass_filter
from model import SENet

# tf.keras.backend.clear_session()


print(tf.__version__)


# Model parameters -----------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--datapath', type=str, default='/home/user/PycharmProjects/SE-Net/Dataset/id_1_data.npz', help='datasets location')
parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=8, help='size of the batches')
parser.add_argument('--foldname', type=str, default='SENet', help='model name')
parser.add_argument('--modelname', type=str, default='/home/user/PycharmProjects/SE-Net/Dataset/SENet.h5', help='File name for store')
opt = parser.parse_args()

ppg_abp_ecg = np.load(opt.datapath)

ppg_sets = ppg_abp_ecg['ppg_data']

abp_sets = ppg_abp_ecg['abp_data']

ecg_sets = ppg_abp_ecg['ecg_data']

scaler = StandardScaler()

ecg = butter_bandpass_filter(scaler.fit_transform(ecg_sets),125)
ppg = butter_bandpass_filter(scaler.fit_transform(ppg_sets),125)

data_total = np.concatenate((ecg,ppg),axis=1)


data_10db = noise_pressoce(data_total,10)
data_15db = noise_pressoce(data_total,15)
data_20db = noise_pressoce(data_total,20)
data_25db = noise_pressoce(data_total,25)
data_30db = noise_pressoce(data_total,30)

all_data = np.concatenate(
    (data_total,
     data_10db,
     data_15db,
     data_20db,
     data_25db,
     data_30db
     ),
     axis=0)


all_bp = np.concatenate(
    (abp_sets,
     abp_sets,
     abp_sets,
     abp_sets,
     abp_sets,
     abp_sets
     ),
     axis=0)

min_value, max_value = MinMax(all_bp)



# gpu_available = tf.test.gpu_device_name()
#
# if gpu_available:
#     print(f"GPU is available. GPU device name: {gpu_available}")
# else:
#     print("No GPU available. TensorFlow will use CPU.")
#
# model = SENet(x=2)
# model.compile(loss=tf.keras.losses.MeanSquaredError(),optimizer='adam',metrics=['mean_squared_error'], loss_weights=[1., 0.9, 0.8, 0.7, 0.6])
#
# model.fit(all_data,  [np.array(all_bp), np.array(min_value), np.array(max_value)],opt.batch_size,opt.n_epochs,verbose=1)






model = SENet(x=2)
model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam', metrics=['mean_squared_error'],
              loss_weights=[1., 0.9, 0.8, 0.7, 0.6])

model.fit(all_data,  [np.array(all_bp), np.array(min_value), np.array(max_value)],opt.batch_size,opt.n_epochs,verbose=1)


model.save(opt.modelname)
