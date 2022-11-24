import os
import sys
import numpy as np
sys.path.append(os.path.join(os.getcwd(), ".."))
from Metrics import Fiducial_metrics,Non_Fiducial_metrics,TSD_cal,Our_SQA_method


dict_functions =  {"Corr_interlead": Fiducial_metrics.Corr_lead_score,
                   "Corr_intralead":Fiducial_metrics.Morph_score,
                   "wPMF":Non_Fiducial_metrics.wPMF_score,
                   "SNRECG" : Non_Fiducial_metrics.SNR_index,
                   "HR":Fiducial_metrics.HR_index_calculator,
                   "Kurtosis" : Fiducial_metrics.Kurto_score,
                   "Flatline":Fiducial_metrics.flatline_score,
                   "SDR":Non_Fiducial_metrics.SDR_score,
                   "TSD" : TSD_cal.TSD_index,
                   "SQA" : Our_SQA_method.SQA_method_score}

list_normalization = ["SNRECG","TSD","Flatline"]

def Wrapper(signals,name_method,fs):
    if name_method not in list(dict_functions.keys()):
        raise ValueError(f"The feature {name_method} is not implemented!")
    elif name_method in list_normalization:
        return dict_functions[name_method](signals,fs,normalization = True)
    else :
        return dict_functions[name_method](signals,fs)

def main(signals,fs,list_methods):
    ##You must give :
    ##1) the ECG leads signals (numpy array [n_leads*len_signals])
    ##2) Sampling frequency (only metadata required)
    ##3) List of method names you want to use
    X = np.zeros([signals.shape[0],len(list_methods)])
    for i in range(len(list_methods)):
        X[:,i] = Wrapper(signals,list_methods[i],fs)

    return X
