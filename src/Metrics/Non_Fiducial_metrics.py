import numpy as np
from scipy.signal import periodogram
import pywt

def SDR_score(signals,fs):
    ##SDR coeff:
    SDR_arr = np.array([])
    for i in range(signals.shape[0]):
        f,PSD = periodogram(signals[i,:],fs)
        QRS_signal_PSD = np.sum(PSD[np.logical_and(f>=5,f<=14)])
        ECG_tot = np.sum(PSD[np.logical_and(f>=5,f<=50)],dtype = np.float64)
        if ECG_tot == 0:
            ECG_tot = np.sum(np.abs(signals[i,:])**2)
            if ECG_tot ==0:
                ECG_tot = 2**63-1
        SDR_val = QRS_signal_PSD/ECG_tot
        SDR_arr = np.append(SDR_arr,SDR_val)
    return SDR_arr


def Wavelet_coef(sig,name,lev):
    All_coeff = pywt.wavedec(sig,name,level = lev)

    CA = All_coeff[0]
    CD = All_coeff[1:len(All_coeff)]
    return CA,CD


def Energy_L2(coeff):
    return np.sum(np.abs(coeff)**2, dtype = np.float64)

def wPMF_score(signals,fs):
    waveletname = 'db4'
    level_w = 9
    wPMF_arr = np.array([],dtype = np.float64)
    for i in range(signals.shape[0]):
        CA_w,CD_w = Wavelet_coef(signals[i,:],waveletname,level_w)
        CD_w = np.array(CD_w,dtype = object)
        CA_w = np.array(CA_w,dtype = object)
        E = np.array([])
        for CD in range(level_w):
            E = np.append(E,Energy_L2(CD_w[-(CD+1)]))
        E = np.append(E,Energy_L2(CA_w[0]))
        Etot = np.sum(E,dtype = np.float64)
        if Etot == 0:
            Etot = Energy_L2(signals[i,:][:int((2**level_w)-1)])
            if Etot ==0:
                Etot = 2**63-1
        p = E/Etot
        SQI_ECG = np.sum(p[3:6])
        wPMF_arr = np.append(wPMF_arr,SQI_ECG)
    return wPMF_arr

def SNR_index(signals,fs):
    SNR_arr = np.array([],dtype = np.float64)
    for i in range(signals.shape[0]):
        f,PSD = periodogram(signals[i,:],fs)
        Sig_PSD = np.sum(PSD[np.logical_and(f>2,f<=40)])
        LF_PSD = np.sum(PSD[np.logical_and(f>=0,f<=2)])
        HF_PSD = np.sum(PSD[np.logical_and(f>40,f<=250)])
        if (LF_PSD+HF_PSD == 0.0):
            SNR = Sig_PSD/(LF_PSD+HF_PSD+0.0001)
        else:
            SNR = Sig_PSD/(LF_PSD+HF_PSD)
        SNR_db = 10*np.log10(SNR)
        if np.isinf(SNR_db):
            SNR_db = -100
        SNR_arr = np.append(SNR_arr,SNR_db)
    return SNR_arr
