import sys
import numpy as np
import os
sys.path.append(os.path.join(os.getcwd(), ".."))
from Metrics import Fiducial_metrics,Non_Fiducial_metrics,TSD_cal
import matplotlib.pyplot as plt


def SQA_SMOTE_method_score(signals,fs):
    ###Scores Index :
    results = np.array([])
    arr_wPMF = Non_Fiducial_metrics.wPMF_score(signals,fs)
    arr_CC = Fiducial_metrics.Corr_lead_score(signals,fs)
    arr_TSD= TSD_cal.TSD_index(signals,fs,normalization = True)
    arr_HR = Fiducial_metrics.HR_index_calculator(signals,fs)
    arr_SNR  = Non_Fiducial_metrics.SNR_index(signals,fs,normalization =True)
    for final in range(signals.shape[0]):
        if arr_HR[final] ==1 :
            val = 7.0019*arr_CC[final]+3.2017*arr_wPMF[final]+3.2948*arr_SNR[final]-11.804*arr_TSD[final]
            val = (np.exp(val))/(1+np.exp(val))
        else :
            val = 0
        results = np.append(results,val)
    return results

def SQA_method_score(signals,fs):
    ###Scores Index :
    arr_CC = Fiducial_metrics.Corr_lead_score(signals,fs)
    arr_TSD= TSD_cal.TSD_index(signals,fs,normalization = True)
    arr_HR = Fiducial_metrics.HR_index_calculator(signals,fs)
    arr_wPMF  = Non_Fiducial_metrics.wPMF_score(signals,fs)
    mean_wPMF = np.mean(arr_wPMF)
    HR_cond = np.min(arr_HR)
    mean_TSD = np.mean(arr_TSD)
    CC_mean = np.mean(arr_CC)
    if HR_cond ==1 :
        val = 5.0540*mean_wPMF-7.6678*mean_TSD+2.8224*CC_mean+3.2312*HR_cond
        val = (np.exp(val))/(1+np.exp(val))
    else:
        val = 0
    return val

# def SQA_Lead_method_score(signals,fs):
#     ###Scores Index :
#     results = np.array([])
#     arr_CC = Fiducial_metrics.Corr_lead_score(signals,fs)
#     arr_TSD= TSD_cal.TSD_index(signals,fs,normalization = True)
#     arr_HR = Fiducial_metrics.HR_index_calculator(signals,fs)
#     arr_wPMF  = Non_Fiducial_metrics.wPMF_score(signals,fs)
#     for final in range(signals.shape[0]):
#         if arr_HR[final] ==1 :
#             val = 5.2395*arr_wPMF [final]-10.093*arr_TSD[final]+6.6979*arr_CC[final]
#             val = (np.exp(val))/(1+np.exp(val))
#         else :
#             val = 0
#         results = np.append(results,val)
#     return results

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
    arr_HR = Fiducial_metrics.HR_index_calculator(signals,fs)
    arr_CC = Fiducial_metrics.Corr_lead_score(signals,fs)
    arr_wPMF  = Non_Fiducial_metrics.wPMF_score(signals,fs)
    mean_wPMF = np.mean(arr_wPMF)
    mean_CC = np.mean(arr_CC)
    HR_index = np.min(arr_HR)
    if HR_index == 1:
        val = 1.0144*mean_wPMF-1.2162*mean_CC+2.4889*mean_wPMF
        val = (np.exp(val))/(1+np.exp(val))
    else :
        val = 0
    return val


# def SQA_Lead_NTSD_method_score(signals,fs):
#     ###Scores Index :
#     results = np.array([])
#     arr_HR = Fiducial_metrics.HR_index_calculator(signals,fs)
#     arr_CC = Fiducial_metrics.Corr_lead_score(signals,fs)
#     arr_SNR  = Non_Fiducial_metrics.SNR_index(signals,fs,normalization = True)
#     for final in range(signals.shape[0]):
#         if arr_HR[final] == 1:
#             val = 0.69726*arr_SNR [final]+2.3343 *arr_CC[final]
#             val = (np.exp(val))/(1+np.exp(val))
#         else :
#             val = 0
#         results = np.append(results,val)
    #eturn results

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

def Model_regularization(signals,fs):

    arr_CC = np.mean(Fiducial_metrics.Corr_lead_score(signals,fs))
    arr_SNR  = np.mean(Non_Fiducial_metrics.SNR_index(signals,fs,normalization = True))
    arr_M = np.mean(Fiducial_metrics.Morph_score(signals,fs))
    arr_wPMF = np.mean(Non_Fiducial_metrics.wPMF_score(signals,fs))
    HR_index = np.min(Fiducial_metrics.HR_index_calculator(signals,fs))
    if HR_index == 1:
        val =  0.4220*arr_SNR+0.4127*arr_CC-4.4785*arr_M+4.5715*arr_wPMF+3.0347*HR_index
        val = (np.exp(val))/(1+np.exp(val))
    else :
        val = 0
    return val
