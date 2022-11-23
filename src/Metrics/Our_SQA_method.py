import sys
import numpy as np
import os
sys.path.append(os.path.join(os.getcwd(), ".."))
from Metrics import Fiducial_metrics,Non_Fiducial_metrics,TSD_cal

def SQA_method_score(signals,fs):
    ###Scores Index :
    results = np.array([])
    #arr_M = Fiducial_metrics.Morph_score(signals,fs)
    arr_wPMF = Non_Fiducial_metrics.wPMF_score(signals,fs)
    arr_CC = Fiducial_metrics.Corr_lead_score(signals,fs)
    arr_TSD= TSD_cal.TSD_index(signals,fs)
    arr_HR = Fiducial_metrics.HR_index_calculator(signals,fs)
    arr_SNR  = Non_Fiducial_metrics.SNR_index(signals,fs)
    for final in range(signals.shape[0]):
        if arr_HR[final] ==1:
            #val = (arr_SNR[final]/arr_TSD[final])*(arr_M[final]*arr_CC[final])
            val = 0.1265*arr_SNR [final]-3.3849*arr_TSD[final]+3.8964*arr_wPMF[final]+5.1442 *arr_CC[final]
        else :
            val = -5
        results = np.append(results,val)
    return results

def SQA_NTSD_method_score(signals,fs):
    ###Scores Index :
    results = np.array([])
    arr_HR = Fiducial_metrics.HR_index_calculator(signals,fs)
    arr_wPMF = Non_Fiducial_metrics.wPMF_score(signals,fs)
    arr_CC = Fiducial_metrics.Corr_lead_score(signals,fs)
    arr_SNR  = Non_Fiducial_metrics.SNR_index(signals,fs)
    for final in range(signals.shape[0]):
        if arr_HR[final] == 1:
            val = 0.3124*arr_SNR [final]-2.8995*arr_wPMF[final]+1.8651 *arr_CC[final]
        else :
            val = -5
        results = np.append(results,val)
    return results
