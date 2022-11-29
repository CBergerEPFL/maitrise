import sys
import numpy as np
import os
sys.path.append(os.path.join(os.getcwd(), ".."))
from Metrics import Fiducial_metrics,Non_Fiducial_metrics,TSD_cal
import matplotlib.pyplot as plt
def SQA_method_score(signals,fs):
    ###Scores Index :
    results = np.array([])
    #arr_M = Fiducial_metrics.Morph_score(signals,fs)
    #arr_wPMF = Non_Fiducial_metrics.wPMF_score(signals,fs)
    arr_CC = Fiducial_metrics.Corr_lead_score(signals,fs)
    arr_TSD= TSD_cal.TSD_index(signals,fs,normalization = True)
    arr_HR = Fiducial_metrics.HR_index_calculator(signals,fs)
    arr_SNR  = Non_Fiducial_metrics.SNR_index(signals,fs,normalization =True)
    for final in range(signals.shape[0]):
        if arr_HR[final] ==1:
            #val = (arr_SNR[final]/arr_TSD[final])*(arr_M[final]*arr_CC[final])
            val = 5.3242*arr_SNR [final]-10.2202*arr_TSD[final]+6.7163 *arr_CC[final]
            val = (np.exp(val))/(1+np.exp(val))
        else :
            val = 0
        results = np.append(results,val)
    return results

def SQA_wrong_estimate(signals,fs,name_lead,y_label,Topt,interval):
    t = np.linspace(0,int(len(signals[0,:])/fs),len(signals[0,:]))
    arr_CC = Fiducial_metrics.Corr_lead_score(signals,fs)
    arr_TSD= TSD_cal.TSD_index(signals,fs,normalization = True)
    arr_HR = Fiducial_metrics.HR_index_calculator(signals,fs)
    arr_SNR  = Non_Fiducial_metrics.SNR_index(signals,fs,normalization =True)
    for final in range(signals.shape[0]):
        val = 5.3242*arr_SNR [final]-10.2202*arr_TSD[final]+6.7163 *arr_CC[final]
        val = (np.exp(val))/(1+np.exp(val))
        if val > Topt:
            prediction = "acceptable"
        else :
            prediction = "unacceptable"
        plt.figure()
        plt.plot(t,signals[final,:].copy())
        plt.title(f"Full signal of Lead {name_lead[final].decode('utf8')}")
        plt.grid()
        plt.xlim(interval)
        plt.figtext(1, 0.8, "HR presence = {0:.2f}".format(arr_HR[final]))
        plt.figtext(1, 0.7, "Normalized SNR value = {0:.2f}".format(arr_SNR [final]))
        plt.figtext(1, 0.6, "Normalized TSD value value = {0:.2f}".format(arr_TSD [final]))
        plt.figtext(1, 0.5, "Intercorrelation value = {0:.2f}".format(arr_CC [final]))
        plt.figtext(1, 0.4, "Logistic Regression prediciton value = {0:.2f}".format(val))
        plt.figtext(1, 0.3, f"Label assigned = {prediction}")
        plt.figtext(1, 0.2, f"True label = {y_label}")

def SQA_NTSD_method_score(signals,fs):
    ###Scores Index :
    results = np.array([])
    arr_HR = Fiducial_metrics.HR_index_calculator(signals,fs)
    arr_Morph = Fiducial_metrics.Morph_score(signals,fs)
    arr_CC = Fiducial_metrics.Corr_lead_score(signals,fs)
    arr_SNR  = Non_Fiducial_metrics.SNR_index(signals,fs,normalization = True)
    for final in range(signals.shape[0]):
        if arr_HR[final] == 1:
            val = 4.5492*arr_SNR [final]+3.9651 *arr_CC[final]-5.6167*arr_Morph[final]
            val = (np.exp(val))/(1+np.exp(val))
        else :
            val = 0
        results = np.append(results,val)
    return results
