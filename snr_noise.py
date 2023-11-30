import numpy as np


def add_noise(signal,SNR):
    Power_sig = (1/len(signal))*np.sum(np.abs(signal)**2,dtype = np.float64)
    P_db = 10*np.log10(Power_sig)
    noisedb = P_db - SNR
    sd_db_watts = 10**(noisedb/10)
    noise = np.random.normal(0,np.sqrt(sd_db_watts),len(signal))
    sig_noisy = signal+noise
    return sig_noisy


def add_observational_noise(sig,SNR):
    Power_sig = (1/len(sig))*np.sum(np.abs(sig)**2,dtype = np.float64)
    P_db = 10*np.log10(Power_sig)
    noisedb = P_db - SNR
    sd_db_watts = 10**(noisedb/10)
    noise = np.random.normal(0,np.sqrt(sd_db_watts),len(sig))
    sig_noisy = sig+noise
    return sig_noisy


def noise_pressoce(data, db):
    data_list = []
    for i in range(0, len(data)):
        test = add_observational_noise(np.array(data[i]), db)
        data_list.append(test)

    return np.array(data_list)