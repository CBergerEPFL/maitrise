from petastorm import make_reader
import numpy as np
import sys
import os
from scipy.signal import periodogram
import matplotlib.pyplot as plt
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
        if (len(cond[cond==True])<0.60*len(signal)):
            return 0
        else :
            print("Warning : Lead ressemble too much ot a flatline ")
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
            print("Warning : Not enough R peaks")
            return 0
        r_sec = x[r_peaks]
        r_msec = r_sec*1000

        RR_bpm_interval = (60/(np.diff(r_msec)))*1000
        mean_RR_interval = np.mean(RR_bpm_interval)
        if mean_RR_interval<24 or mean_RR_interval>450:
            print("Warning : Non pyhiological Heart Rate determinable from this lead ")
            return 2
        else :
            return mean_RR_interval/np.std(RR_bpm_interval)

    def HR_score_dico(self,name_lead):
        array_results = {}
        for i in name_lead:
            array_results[i] = SQA_method.HR_score(self,signal = self.dico_ECG[i])
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
            print("Warning :  No QRS template determinable from this lead ")
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

    def Corr_lead_score(self,name_lead):
    # dico_template = {}
        dico_results = {}
        copy_name = name_lead.copy()
        X = np.empty([len(copy_name),len(self.dico_ECG[copy_name[0]])])
        for i in range(len(copy_name)):
            X[i,:] = self.dico_ECG[copy_name[i]]
        M = np.corrcoef(X)
        if M.size == 1:
            dico_results[name_lead[0]] = (1,self.dico_ECG[name_lead[0]])
            return dico_results,1/12
        M_arr = np.array([])
        for j in range(len(copy_name)):
            val = (np.mean(np.abs(M[j,:])))
            dico_results[copy_name[j]] = (val,self.dico_ECG[copy_name[j]])
            M_arr = np.append(M_arr,val)
        return dico_results,np.mean(M_arr)

    def SNR_index(self,name_lead):
        SNR_lead = {}
        SNR_arr = np.array([],dtype = np.float64)
        for i in name_lead:
            f,PSD = periodogram(self.dico_ECG[i],self.fs)
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
            SNR_lead[i] = (SNR_db,self.dico_ECG[i])
            SNR_arr = np.append(SNR_arr,SNR_db)
        return SNR_lead,np.mean(SNR_arr)

    def Corr_dico_score(self,name_lead):
        results = np.array([])
        dic,_  = SQA_method.Corr_lead_score(self,name_lead)
        for i in name_lead:
            if dic[i][0]>=0.40:
                results = np.append(results,1)
            else :
                results = np.append(results,0)
        return results
    def Morph_dico_score(self,name_lead):
        array_results = np.array([])
        for i in name_lead:
            res = SQA_method.Morph_sig_score(self,self.dico_ECG[i])
            if res>=0.66:
                array_results = np.append(array_results,1)
            else :
                array_results = np.append(array_results,0)
        return array_results

    def SQA_method_score(self):
    ###Scores Index :
        dico_results = {}
        copy_name = self.ECG_lead.copy()
        Dico_M,_ = SQA_method.Morph_dico_score(self,copy_name)
        Dico_CC,_ = SQA_method.Corr_lead_score(self,copy_name)
        Dico_TSD,_= TSD.TSD_index(self.dico_ECG,copy_name,self.fs)
        Dico_HR = SQA_method.HR_score_dico(self,copy_name)
        Dico_SNR,_ = SQA_method.SNR_index(self,copy_name)
        for final in copy_name:
            val = (Dico_SNR[final][0]/Dico_HR[final][0])-Dico_TSD[final][0]*(Dico_M[final][0]*Dico_CC[final][0])
            dico_results[final] = (val,self.dico_ECG[final])
        return dico_results

    def comparative_plot_TSD_signal(self,dico_resultat):
        t = SQA_method.get_time_axis(self)

        for i in self.ECG_lead:
            print(dico_resultat[i][2])
            Dl,Ds,_ = TSD.TSD_calculator(dico_resultat[i][1],100,self.fs)
            plt.figure()
            _,ax = plt.subplots(nrows = 2, ncols=2,figsize=(20,15))
            ax[0,0].plot(t,dico_resultat[i][1].copy(),label = "TSD score = {0:.2f}".format(Ds))
            ax[0,0].set_title(f"Time Evolution of Lead {i.decode('utf8')}")
            ax[0,0].set_xlabel("Time (sec)")
            ax[0,0].set_ylabel("Volatge Amplitude")
            ax[0,0].legend()
            ax[0,0].grid()
            ax[1,0].plot(np.linspace(0,int(len(Dl)/self.fs),len(Dl)),Dl,label = "TSD score = {0:.2f}".format(Ds))
            ax[1,0].plot(t,np.ones_like(t)*Ds,"--k",label = "Mean TSD value")
            ax[1,0].set_title(f"TSD evolution for {i.decode('utf8')}  using a segment length of {100}")
            ax[1,0].grid()
            ax[1,0].legend()
            ax[1,0].set_xlabel("Lags")
            ax[1,0].set_ylabel("TSD value")
            ax[0,1].plot(t,dico_resultat[i][1].copy(),label = "TSD score = {0:.2f}".format(Ds))
            ax[0,1].set_title(f"Time Evolution of Lead {i.decode('utf8')} for an time interval of {[0,3]}")
            ax[0,1].set_xlabel("Time (sec)")
            ax[0,1].set_ylabel("Volatge Amplitude")
            ax[0,1].set_xlim([0,3])
            ax[0,1].legend()
            ax[0,1].grid()
            ax[1,1].plot(np.linspace(0,int(len(Dl)/self.fs),len(Dl)),Dl,label = "TSD score = {0:.2f}".format(Ds))
            ax[1,1].set_title(f"TSD evolution for {i.decode('utf8')}  using a segment length of {100} for a time interval of {[0,3]}")
            ax[1,1].grid()
            ax[1,1].plot(t,np.ones_like(t)*Ds,"--k",label = "Mean TSD value")
            ax[1,1].legend()
            ax[1,1].set_xlim([0,3])
            ax[1,1].set_xlabel("Lags")
            ax[1,1].set_ylabel("TSD value")
            plt.show()
