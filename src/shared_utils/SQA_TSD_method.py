from petastorm import make_reader
import numpy as np
import sys
import os
from scipy.stats import pearsonr
from ecgdetectors import Detectors
sys.path.append(os.path.join(os.getcwd(), ".."))
from shared_utils import TSD_cal as TSD


class SQA_method():
    def __init__(self,path_data):
        with make_reader(path_data) as reader:
            for sample in reader:
                data = sample
                break
        self.ECG_lead = data.signal_names
        self.fs = data.sampling_frequency
        self.dico_ECG = {}
        for i,j in zip(self.ECG_lead,range(12)):
            self.dico_ECG[i] = data.signal[:,j]
        self.N = len(self.dico_ECG[self.ECG_lead[0]])
        self.detect = Detectors(self.fs)
        self.patient_name = data.noun_id


    def get_time_axis(self):
        x = np.linspace(0,int(self.N/self.fs),self.N)
        return x

    def flatline_score(self,signal):
        cond = np.where(np.diff(signal)!=0.0,False,True)
        if (len(cond[cond==True])<0.45*len(signal)):
            return 0
        else :
            return len(cond[cond==True])/len(signal)

    def Flatline_dico_score(self,name_lead):
        array_results = np.array([])
        for i in name_lead:
            if SQA_method.flatline_score(self,self.dico_ECG[i])>0:
                array_results = np.append(array_results,0)
            else :
                array_results = np.append(array_results,1)
        return array_results

    def HR_score(self,signal):
        mean_RR_interval = 0
        x = SQA_method.get_time_axis(self)
        r_peaks = self.detect.pan_tompkins_detector(signal)
        if len(r_peaks)<=2:
            return 0
        r_sec = x[r_peaks]
        r_msec = r_sec*1000

        RR_bpm_interval = (60/(np.diff(r_msec)))*1000
        mean_RR_interval = np.mean(RR_bpm_interval)
        if mean_RR_interval<24 or mean_RR_interval>450:
            return 0
        else :
            return 1

    def HR_score_dico(self,name_lead):
        array_results = np.array([])
        for i in name_lead:
            res = SQA_method.HR_score(self,signal = self.dico_ECG[i])
            array_results = np.append(array_results,res)
        return array_results

    def PQRST_template_extractor(self,ECG_signal,rpeaks):
    ##From the Biosspy function _extract_heartbeats
        R = np.sort(rpeaks)
        length = len(ECG_signal)
        templates = []
        newR = []

        for r in R:
            a = r-(np.median(np.diff(rpeaks,1))/2)
            if a < 0:
                continue
            b = r+(np.median(np.diff(rpeaks,1))/2)
            if b > length:
                break

            templates.append(ECG_signal[int(a):int(b)])
            newR.append(r)

        templates = np.array(templates)
        newR = np.array(newR, dtype="int")

        return templates, newR

    def Morph_sig_score(self,signal):
    ##SDR coeff:
        r_peaks = self.detect.pan_tompkins_detector(signal)
        template,_ = SQA_method.PQRST_template_extractor(self,signal,rpeaks = r_peaks)
        empty_index = np.array([],dtype = int)
        for ble in range(template.shape[0]):
            if template[ble].size==0:
                empty_index = np.append(empty_index,ble)
        template = np.delete(template,empty_index,0)
        index_maxima = np.array([np.argmax(template[w]) for w in range(template.shape[0])])
        median_index = np.median(index_maxima.copy())
        templates_good = template[np.isclose(index_maxima.copy(),median_index,rtol=0.1)].copy()
        if templates_good.size == 0:
            return 0
        sig_mean = templates_good[0]
        for i in range(1,templates_good.shape[0]):
            if sig_mean.size != templates_good[i].size:
                templates_good[i] = templates_good[i][:len(sig_mean)]
            sig_mean = np.add(sig_mean,templates_good[i].copy())

        sig = sig_mean/len(templates_good)
        r_p = np.array([])
        for t in templates_good:
            r_p = np.append(r_p,pearsonr(sig,t)[0])

        return np.mean(r_p)

    def Morph_dico_score(self,name_lead):
        array_results = np.array([])
        for i in name_lead:
            res = SQA_method.Morph_sig_score(self,self.dico_ECG[i])
            if res>=0.5:
                array_results = np.append(array_results,1)
            else :
                array_results = np.append(array_results,0)
        return array_results

    def SQA_method_score(self):
    ###Scores Index :
        dico_results = {}
        copy_name = self.ECG_lead.copy()

    #3 stages check.1st : Flatlines:
        flatline_lead = SQA_method.Flatline_dico_score(self,copy_name)
        if not flatline_lead.all():
            flat_lead = copy_name[flatline_lead ==0]
            for f in flat_lead:
                dico_results[f] = (2,self.dico_ECG[f])
            copy_name = copy_name[flatline_lead !=0]
        if len(copy_name) == 0:
            return dico_results
    ##Second : HR value we got:
        HR_lead = SQA_method.HR_score_dico(self,copy_name)

        if not HR_lead.all():
            HR_bad_lead  = copy_name[HR_lead == 0]
            for h in HR_bad_lead:
                dico_results[h] = (2,self.dico_ECG[h])
            copy_name = copy_name[HR_lead!=0]
        if len(copy_name) == 0:
            return dico_results
    ###3rd : Template Matching:
        TM_lead = SQA_method.Morph_dico_score(self,copy_name)
        if not TM_lead.all():
            TM_bad_lead  = copy_name[TM_lead == 0]
            for t in TM_bad_lead:
                dico_results[t] = (2,self.dico_ECG[t])
            copy_name = copy_name[TM_lead!=0]
        if len(copy_name) == 0:
            return dico_results
        Dico_TSD,_=TSD.TSD_index(self.dico_ECG,copy_name,self.fs)
        for final in copy_name:
            dico_results[final] = ((Dico_TSD[final][0]),self.dico_ECG[final])
        return dico_results
