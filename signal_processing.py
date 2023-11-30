import os
import argparse
import numpy as np
import mat73



# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1, 2, 3'

# Model parameters -----------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--datapath', type=str, default='/home/user/data_bp/raw_data/', help='datasets')
parser.add_argument('--part_id', type=int, default=2, help='part id')
parser.add_argument('--ppg_data', type=str, default='ppg_data.pickle', help='save ppg data datasets')
parser.add_argument('--abp_data', type=str, default='abp_data.pickle', help='save abp data datasets')
parser.add_argument('--ecg_data', type=str, default='ecg_data.pickle', help='save ecg data datasets')
parser.add_argument('--cachefile', type=str, default='/home/user/PycharmProjects/SE-Net/Dataset/id_1_data.npz', help='File name for store')
opt = parser.parse_args()


def data_segment(data, size):
    ecg = []
    bp = []
    ppg = []

    for i in range(0, 300): # 样本个数设置
        temp_mat = data[i]
        temp_length = temp_mat.shape[1]
        for j in range((int)(temp_length / size)):
            temp_ecg = temp_mat[2, j * size:(j + 1) * size]
            temp_abp = temp_mat[1, j * size:(j + 1) * size]
            temp_ppg = temp_mat[0, j * size:(j + 1) * size]

            ecg.append(temp_ecg)
            bp.append(temp_abp)
            ppg.append(temp_ppg)

    return ecg, ppg, bp


def process_batch(signals_batch):

    del_inds = {}
    for i, signal in enumerate(signals_batch):

        signals_batch[i] = signal
        segment_mean = np.mean(signal, axis=0)
        errors = np.abs(signal - segment_mean)
        std_dev = np.std(errors, axis=0)
        anomaly_indices = np.where(errors > std_dev + segment_mean)
        del_inds[i] = anomaly_indices[0].tolist()

    return np.array(signals_batch), del_inds


def fill_data(signals_batch):
    for i, signal in enumerate(signals_batch):
        if len(signal) >= 250:
            signal = signal[:250]
        if len(signal) <= 250:
            padding = 250 - len(signal)
            signal = np.pad(signal, (0, padding), 'constant', constant_values=0)

        signals_batch[i] = signal

    return signals_batch

def del_noise(ECGs,PPGs,ABPs):

    ECGs, ECG_inds = process_batch(ECGs)
    PPGs, PPG_inds = process_batch(PPGs)
    ABPs, ABP_inds = process_batch(ABPs)

    ppg_data = []
    ecg_data = []
    abp_data = []

    for i in range(len(PPGs)):
        #     rmv = list(set(PPG_inds[i] + ABP_inds[i]))
        # Get unique anomalies combined from each signal
        rmv = list(set(PPG_inds[i] + ECG_inds[i] + ABP_inds[i]))

        newPPG = np.delete(PPGs[i], rmv, axis=0)
        newECG = np.delete(ECGs[i], rmv, axis=0)
        newABP = np.delete(ABPs[i], rmv, axis=0)


        ppg_data.append(newPPG)
        ecg_data.append(newECG)
        abp_data.append(newABP)


    return ppg_data,ecg_data,abp_data


#
# def butter_bandpass(lowcut, highcut, sRate, order=2):
#     nyq = 0.5 * sRate
#     low = lowcut / nyq
#     high = highcut / nyq
#     b, a = butter(order, [low, high], btype='band')
#     return b, a
# def butter_bandpass_filter(data, lowcut, highcut, sRate, order=2):
#     b, a = butter_bandpass(lowcut, highcut, sRate, order=order)
#     zi = lfilter_zi(b, a)
#     y,zo = lfilter(b, a, data, zi=zi*data[0])
#     return y



def Extracting_data():
    lowcut = 0.9
    highcut = 10
    order = 2
    sps = 100

    data_mat = mat73.loadmat(opt.datapath + '/Part_{}.mat'.format(opt.part_id))
    data = data_mat['Part_{}'.format(opt.part_id)]
    ECGs,PPGs,ABPs = data_segment(data,270)
    ppg_data,ecg_data,abp_data = del_noise(ECGs,PPGs,ABPs)
    ppg = fill_data(ppg_data)
    ecg = fill_data(ecg_data)
    abp = fill_data(abp_data)


    np.savez(opt.cachefile, ecg_data=ecg, ppg_data=ppg,abp_data =abp) # Save cahce file

    print('well done')




