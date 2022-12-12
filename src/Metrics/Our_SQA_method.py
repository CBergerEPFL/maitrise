import sys
import numpy as np
import os
import pickle
import pandas as pd
import warnings
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.getcwd(), ".."))
from Metrics import Fiducial_metrics,Non_Fiducial_metrics,TSD_cal
warnings.simplefilter(action="ignore", category=FutureWarning)
dict_functions = {
    "Corr_interlead": Fiducial_metrics.Corr_lead_score,
    "Corr_intralead": Fiducial_metrics.Morph_score,
    "wPMF": Non_Fiducial_metrics.wPMF_score,
    "SNRECG": Non_Fiducial_metrics.SNR_index,
    "HR": Fiducial_metrics.HR_index_calculator,
    "Kurtosis": Fiducial_metrics.Kurto_score,
    "Flatline": Fiducial_metrics.flatline_score,
    "SDR": Non_Fiducial_metrics.SDR_score,
    "TSD": TSD_cal.TSD_index,
}

folder_model_path = "/workspaces/maitrise/results/Models"
save_path = "/workspaces/maitrise/results"
feat_SQA = ["TSD","Corr_interlead","HR","SNRECG"]
feat_SQANOTSD = ["Corr_interlead","HR","SNRECG"]
feat_L2Reg = ['Corr_interlead', 'Corr_intralead', 'wPMF', 'SNRECG', 'HR']
feat_JMI_MI = ["Corr_interlead","HR","SNRECG","Corr_intralead"]

name_SQA_opp_model =  "Logit_bin_TSD_Corr_interlead_HR_SNRECG_inverselabel"
name_SQA_model =  "Logit_bin_TSD_Corr_interlead_HR_SNRECG_"
name_NTSDSQA_opp_model =  "Logit_bin_Corr_interlead_HR_SNRECG_inverselabel"
name_NTSDSQA_model =  "Logit_bin_Corr_interlead_HR_SNRECG_"
name_L2_model =  "Logit_bin_Corr_interlead_Corr_intralead_wPMF_SNRECG_HR_"
name_L2_opp_model =  "Logit_bin_Corr_interlead_Corr_intralead_wPMF_SNRECG_HR_inverselabel"
name_JMI_MI_opp_model = "Logit_bin_Corr_interlead_HR_SNRECG_Corr_intralead_inverselabel"
name_JMI_MI_model = "Logit_bin_Corr_interlead_HR_SNRECG_Corr_intralead_"



def SQA_method_score(signals,fs,**kwargs):

    if not os.path.exists(folder_model_path):
        os.mkdir(folder_model_path)
        raise AttributeError("Please, have your model already trained and ready to go!")

    if kwargs.get("opposite"):
        if kwargs["opposite"]:
            name = name_SQA_opp_model
        else :
            name = name_SQA_model
    else :
        name= name_SQA_model

    model = pickle.load(open(os.path.join(folder_model_path,name+".sav"), 'rb'))
    X_test = np.empty(len(feat_SQA))
    for count,name in enumerate(feat_SQA):
        if name == "HR":
            X_test[count] = np.min(dict_functions[name](signals,fs))
        else:
            X_test[count] = np.mean(dict_functions[name](signals,fs,normalization = True))
    X_test = X_test.reshape(1, -1)
    y_proba = model.predict_proba(X_test)
    return y_proba[:,1]

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

def SQA_NTSD_method_score(signals,fs,**kwargs):
    if not os.path.exists(folder_model_path):
        os.mkdir(folder_model_path)
        raise AttributeError("Please have your model trained and saved!")

    if kwargs.get("opposite"):
        if kwargs["opposite"]:
            name = name_NTSDSQA_opp_model
        else :
            name = name_NTSDSQA_model
    else :
        name= name_NTSDSQA_model
    model = pickle.load(open(os.path.join(folder_model_path,name+".sav"), 'rb'))
    X_test = np.empty([len(feat_SQANOTSD)])
    for count,name in enumerate(feat_SQANOTSD):
        if name == "HR":
            X_test[count] = np.min(dict_functions[name](signals,fs))
        else:
            X_test[count] = np.mean(dict_functions[name](signals,fs,normalization = True))
    X_test = X_test.reshape(1, -1)
    y_proba = model.predict_proba(X_test)
    return y_proba[:,1]



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

def Model_regularization(signals,fs,**kwargs):

    if not os.path.exists(folder_model_path):
        os.mkdir(folder_model_path)
        raise AttributeError("Please have your model trained and saved!")

    if kwargs.get("opposite"):
        if kwargs["opposite"]:
            name = name_L2_opp_model
        else :
            name = name_L2_model
    else :
        name= name_L2_model

    model = pickle.load(open(os.path.join(folder_model_path,name+".sav"), 'rb'))
    X_test = np.empty([len(feat_L2Reg)])
    for count,name in enumerate(feat_L2Reg):
        if name == "HR":
            X_test[count] = np.min(dict_functions[name](signals,fs))
        else:
            X_test[count] = np.mean(dict_functions[name](signals,fs,normalization = True))
    X_test = X_test.reshape(1, -1)
    y_proba = model.predict_proba(X_test)
    return y_proba[:,1]

def Model_MI(signals,fs,**kwargs):

    if not os.path.exists(folder_model_path):
        os.mkdir(folder_model_path)
        raise AttributeError("Please have your model trained and saved!")

    if kwargs.get("opposite"):
        if kwargs["opposite"]:
            name = name_JMI_MI_opp_model
        else :
            name = name_JMI_MI_model
    else :
        name= name_JMI_MI_model

    model = pickle.load(open(os.path.join(folder_model_path,name+".sav"), 'rb'))
    X_test = np.empty([len(feat_JMI_MI)])
    for count,name in enumerate(feat_JMI_MI):
        if name == "HR":
            X_test[count] = np.min(dict_functions[name](signals,fs))
        else:
            X_test[count] = np.mean(dict_functions[name](signals,fs,normalization = True))
    X_test = X_test.reshape(1, -1)
    y_proba = model.predict_proba(X_test)
    return y_proba[:,1]
