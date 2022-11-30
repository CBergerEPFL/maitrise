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
    arr_CC = Fiducial_metrics.Corr_lead_score(signals,fs)
    arr_TSD= TSD_cal.TSD_index(signals,fs,normalization = True)
    arr_HR = Fiducial_metrics.HR_index_calculator(signals,fs)
    arr_SNR  = Non_Fiducial_metrics.SNR_index(signals,fs,normalization =True)
    for final in range(signals.shape[0]):
        if arr_HR[final] ==1 :
            val = 5.2396*arr_SNR [final]-10.0932*arr_TSD[final]+6.6980  *arr_CC[final]
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
    #arr_wPMF = Non_Fiducial_metrics.wPMF_score(signals,fs)
    for final in range(signals.shape[0]):
        val = 5.2396*arr_SNR [final]-10.0932*arr_TSD[final]+6.6979 *arr_CC[final]
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
    arr_CC = Fiducial_metrics.Corr_lead_score(signals,fs)
    arr_SNR  = Non_Fiducial_metrics.SNR_index(signals,fs,normalization = True)
    for final in range(signals.shape[0]):
        if arr_HR[final] == 1:
            val = 0.69726*arr_SNR [final]+2.3343 *arr_CC[final]
            val = (np.exp(val))/(1+np.exp(val))
        else :
            val = 0
        results = np.append(results,val)
    return results

def SQANTSD_wrong_estimate(signals,fs,name_lead,y_label,Topt,interval):
    t = np.linspace(0,int(len(signals[0,:])/fs),len(signals[0,:]))
    arr_CC = Fiducial_metrics.Corr_lead_score(signals,fs)
    arr_HR = Fiducial_metrics.HR_index_calculator(signals,fs)
    arr_SNR  = Non_Fiducial_metrics.SNR_index(signals,fs,normalization =True)
    #arr_wPMF = Non_Fiducial_metrics.wPMF_score(signals,fs)
    for final in range(signals.shape[0]):
        val = 0.69726*arr_SNR [final]+2.3343 *arr_CC[final]
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
        plt.figtext(1, 0.6, "Intercorrelation value = {0:.2f}".format(arr_CC [final]))
        #plt.figtext(1, 0.5, "wPMF value = {0:.2f}".format(arr_wPMF [final]))
        plt.figtext(1, 0.5, "Logistic Regression prediciton value = {0:.2f}".format(val))
        plt.figtext(1, 0.4, f"Label assigned = {prediction}")
        plt.figtext(1, 0.3, f"True label = {y_label}")


def Model_autoFeatureSel_SMOTED(signals,fs):
    results = np.array([])
    arr_HR = Fiducial_metrics.HR_index_calculator(signals,fs)
    arr_CC = Fiducial_metrics.Corr_lead_score(signals,fs)
    arr_SNR  = Non_Fiducial_metrics.SNR_index(signals,fs,normalization = True)
    arr_wPMF = Non_Fiducial_metrics.wPMF_score(signals,fs)
    for final in range(signals.shape[0]):
        if arr_HR[final] == 1:
            val = -1.0402*arr_SNR [final]+2.1715 *arr_CC[final]+3.3366*arr_wPMF[final]
            val = (np.exp(val))/(1+np.exp(val))
        else :
            val = 0
        results = np.append(results,val)
    return results

def Model_autoFeatureSel(signals,fs):
    results = np.array([])
    arr_HR = Fiducial_metrics.HR_index_calculator(signals,fs)
    arr_CC = Fiducial_metrics.Corr_lead_score(signals,fs)
    arr_SNR  = Non_Fiducial_metrics.SNR_index(signals,fs,normalization = True)
    arr_M = Fiducial_metrics.Morph_score(signals,fs)
    for final in range(signals.shape[0]):
        if arr_HR[final] == 1:
            val =  4.4117*arr_SNR [final]+3.9355*arr_CC[final]-5.4417*arr_M[final]
            val = (np.exp(val))/(1+np.exp(val))
        else :
            val = 0
        results = np.append(results,val)
    return results
