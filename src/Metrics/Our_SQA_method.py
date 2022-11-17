import sys
import numpy as np
import os
sys.path.append(os.path.join(os.getcwd(), ".."))
from Metrics import Fiducial_metrics,Non_Fiducial_metrics,TSD_cal

def SQA_method_score(signals,fs):
    ###Scores Index :
    results = np.array([])
    arr_M = Fiducial_metrics.Morph_score(signals,fs)
    arr_CC = Fiducial_metrics.Corr_lead_score(signals,fs)
    arr_TSD= TSD_cal.TSD_index(signals,fs)
    arr_HR = Fiducial_metrics.HR_index_calculator(signals,fs)
    arr_SNR  = Non_Fiducial_metrics.SNR_index(signals,fs)
    for final in range(signals.shape[0]):
        val = (arr_SNR[final]/arr_HR[final])-arr_TSD[final]*(arr_M[final][0]*arr_CC[final][0])
        results = np.append(results,val)
    return results

def SQA_NTSD_method_score(signals,fs):
    ###Scores Index :
    results = np.array([])
    arr_M = Fiducial_metrics.Morph_score(signals,fs)
    arr_CC = Fiducial_metrics.Corr_lead_score(signals,fs)
    arr_TSD= TSD_cal.TSD_index(signals,fs)
    arr_HR = Fiducial_metrics.HR_index_calculator(signals,fs)
    arr_SNR  = Non_Fiducial_metrics.SNR_index(signals,fs)
    for final in range(signals.shape[0]):
        val = (arr_SNR[final]/arr_HR[final])
        results = np.append(results,val)
    return results
