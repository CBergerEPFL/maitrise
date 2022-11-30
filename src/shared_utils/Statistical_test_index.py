from petastorm import make_reader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.widgets import TextBox, Button
import sys
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score,auc,precision_score,recall_score
from sklearn.model_selection import StratifiedKFold
import xarray as xr
sys.path.append(os.path.join(os.getcwd(), ".."))
from Metrics import Our_SQA_method
import shared_utils.utils_data as utils_data

save_path = "/workspaces/maitrise/results"
class Statistic_reader():
    def __init__(self,path_peta,function,name_function,Threshold,cv_k = 10,opp = False,**kwargs):
        self.alternate = opp

        with make_reader(path_peta) as reader:
            for sample in reader:
                data = sample
                self.ECG_lead = sample.signal_names
                self.fs = sample.sampling_frequency
                break

        if not "ecg_data.nc" in os.listdir(save_path):
            ds_data = utils_data.format_data_to_xarray(path_peta, save_path)
            ds_filtered = ds_data.where(ds_data.data_quality != "unlabeled").dropna(dim = "id")
            self.Data = ds_filtered.signal.values
            self.names = ds_filtered.id.values.astype(int)
            self.y = ds_filtered.data_quality.values
        else:
            ds_data = xr.load_dataset(os.path.join(save_path,"ecg_data.nc"))
            ds_filtered = ds_data.where(ds_data.data_quality != "unlabeled").dropna(dim = "id")
            self.Data = ds_filtered.signal.values
            self.names = ds_filtered.id.astype(int)
            self.y = ds_filtered.data_quality.values

        if kwargs.get("normalization") :
            self.norma = True
        else :
            self.norma = False
        if kwargs.get("evaluation"):
            self.eval = kwargs["evaluation"]
        else :
            self.eval = ""

        self.function = function
        self.name_f = name_function
        self.k = cv_k

        self.T = np.linspace(Threshold[0],Threshold[1],500)

        self.F_score_train = np.empty([self.k,len(self.T)])
        self.F_score_test = np.empty([self.k,len(self.T)])
        self.Acc_score_train = np.empty([self.k,len(self.T)])
        self.Acc_score_test = np.empty([self.k,len(self.T)])
        self.Prec_score_train = np.empty([self.k,len(self.T)])
        self.Prec_score_test = np.empty([self.k,len(self.T)])
        self.Recall_score_train = np.empty([self.k,len(self.T)])
        self.Recall_score_test = np.empty([self.k,len(self.T)])
        self.FPR_score_train = np.empty([self.k,len(self.T)])
        self.FPR_score_test = np.empty([self.k,len(self.T)])
        self.TPR_score_train = np.empty([self.k,len(self.T)])
        self.TPR_score_test = np.empty([self.k,len(self.T)])
        self.auc_train_ROC = np.array([])
        self.auc_test_ROC = np.array([])
        self.auc_train_PR = np.array([])
        self.auc_test_PR = np.array([])
        self.T_opt_train = np.array([])
        self.T_opt_test = np.array([])
        self.ix_tr = 0
        self.ix_t = 0

    def to_labels(self,pos_probs, threshold):

        if self.norma:
            return (pos_probs >= threshold).astype('int')

        else :
            if not self.alternate:
                return (pos_probs >= threshold).astype('int')
            else :
                return (pos_probs <= threshold).astype('int')

    def create_dataset(self):
        X_data = np.array([])
        if self.norma:
            for x in range(self.Data.shape[0]):
                val = self.function(self.Data[x,:,:].T,self.fs,normalization = True)
                if self.eval == "minimum":
                    X_data = np.append(X_data,np.min(val))
                else :
                    X_data = np.append(X_data,np.mean(val))
        else :
            for x in range(self.Data.shape[0]):
                val = self.function(self.Data[x,:,:].T,self.fs)
                if self.eval == "minimum":
                    X_data = np.append(X_data,np.min(val))
                else :
                    X_data = np.append(X_data,np.mean(val))
        return X_data


    def roc_pr_curve(self,y_true, y_prob):
        fpr = []
        tpr = []
        prec = []
        rec = []
        for threshold in self.T:
            if self.norma :
                y_pred = np.where(y_prob>=threshold,1,0)
            else :
                if not self.alternate:
                    y_pred = np.where(y_prob >= threshold, 1, 0)
                else :
                    y_pred = np.where(y_prob <= threshold, 1, 0)

            fp = np.sum((y_pred == 1) & (y_true == 0))
            tp = np.sum((y_pred == 1) & (y_true == 1))

            fn = np.sum((y_pred == 0) & (y_true == 1))
            tn = np.sum((y_pred == 0) & (y_true == 0))

            if fp == 0:
                prec.append(1)
                rec.append(tp / (tp + fn))
            elif tp == 0:
                prec.append(0)
                rec.append(0)
            else :
                prec.append(tp / (fp + tp))
                rec.append(tp / (tp + fn))
            fpr.append(fp / (fp + tn))
            tpr.append(tp / (tp + fn))

        return fpr,tpr,prec,rec

    def CrossValidation_index_opt_thresh(self):
        cv = StratifiedKFold(n_splits=self.k, random_state=1, shuffle=True)
        ind = 0
        #indexation = np.array(list(range(len(self.names)))).astype(int)
        X_data = Statistic_reader.create_dataset(self)
        y = self.y.copy()
        y[y=="acceptable"] = 1
        y[y=="unacceptable"] = 0
        y = y.astype(int)
        for train_index, test_index in cv.split(X_data,y):
            X_train,X_test = X_data[train_index].copy(),X_data[test_index].copy()
            y_train,y_test = y[train_index].copy(),y[test_index].copy()

            F_train = [f1_score(y_train, Statistic_reader.to_labels(self,X_train, t)) for t in self.T]
            F_test = [f1_score(y_test, Statistic_reader.to_labels(self,X_test, t)) for t in self.T]
            Acc_train = [accuracy_score(y_train,Statistic_reader.to_labels(self,X_train,t)) for t in self.T]
            Acc_test = [accuracy_score(y_test,Statistic_reader.to_labels(self,X_test,t)) for t in self.T]
            fpr_train,tpr_train,prec_train,rec_train = Statistic_reader.roc_pr_curve(self,y_train,X_train)
            fpr_test,tpr_test,prec_test,rec_test = Statistic_reader.roc_pr_curve(self,y_test,X_test)

            self.F_score_train[ind,:] = F_train
            self.F_score_test[ind,:] = F_test
            self.TPR_score_train[ind,:] = tpr_train
            self.TPR_score_test[ind,:] = tpr_test
            self.FPR_score_train[ind,:] = fpr_train
            self.FPR_score_test[ind,:] = fpr_test
            self.Acc_score_train[ind,:] = Acc_train
            self.Acc_score_test[ind,:] = Acc_test
            self.Prec_score_train[ind,:]= prec_train
            self.Prec_score_test[ind,:]= prec_test
            self.Recall_score_train[ind,:]= rec_train
            self.Recall_score_test[ind,:]= rec_test
            self.T_opt_train = np.append(self.T_opt_train,self.T[np.argmax(F_train)])
            self.T_opt_test = np.append(self.T_opt_test,self.T[np.argmax(F_test)])
            ind +=1


    def Optimal_threshold_calculator(self):
        F1_train_mean = self.F_score_train.mean(axis=0)#np.array([np.mean(self.F_score_train[:,j]) for j in range(self.F_score_train.shape[1])])
        F1_train_sd = self.F_score_train.std(axis=0)#np.array([np.std(self.F_score_train[:,j]) for j in range(self.F_score_train.shape[1])])
        F1_test_mean = self.F_score_test.mean(axis=0)#np.array([np.mean(self.F_score_test[:,j]) for j in range(self.F_score_test.shape[1])])
        F1_test_sd = self.F_score_test.std(axis=0)#np.array([np.std(self.F_score_test[:,j]) for j in range(self.F_score_test.shape[1])])
        ix_train = np.argmax(F1_train_mean)
        ix_test = np.argmax(F1_test_mean)
        _,ax = plt.subplots(nrows = 2,ncols = 1,figsize = (15,15))
        ax[0].set_title(f"F1 score curve from {self.k} fold CV training set")
        ax[0].set_xlabel("Threshold")
        ax[0].set_ylabel("F1-score")
        ax[0].grid()
        ax[0].scatter(self.T[ix_train],F1_train_mean[ix_train], marker='o', color='black', label='Best')
        ax[0].errorbar(self.T,F1_train_mean,yerr = F1_train_sd)
        ax[0].legend(loc = 4)
        ax[1].set_title(f"F1 score curve from {self.k} fold CV testing set")
        ax[1].set_xlabel("Threshold")
        ax[1].set_ylabel("F1-score")
        ax[1].scatter(self.T[ix_test],F1_test_mean[ix_test], marker='o', color='black', label='Best')
        ax[1].errorbar(self.T,F1_test_mean,yerr = F1_test_sd)
        ax[1].grid()
        ax[1].legend(loc = 4)
        print("Best Threshold for {} dataset of {} : ".format("Training",self.name_f), self.T[ix_train], "with F1-score : {} +- {}".format(np.max(F1_train_mean),F1_train_sd[np.argmax(F1_train_mean)]))
        print("Best Threshold for {} dataset of {} : ".format("Testing",self.name_f), self.T[ix_test], "with F1-score : {} +- {}".format(np.max(F1_test_mean),F1_test_sd[np.argmax(F1_test_mean)]))
        print("From Training F1 curve : T_optimal = ",self.T[ix_train])
        print("From Testing F1 curve : T_optimal = ",self.T[ix_test])
        self.ix_tr = ix_train
        self.ix_t = ix_test

    def plot_ROC_curve(self):
        TPR_train_mean = self.TPR_score_train.mean(axis = 0)#np.array([np.mean(self.TPR_score_train[:,j]) for j in range(self.TPR_score_train.shape[1])])
        TPR_train_sd = self.TPR_score_train.std(axis = 0)#np.array([np.std(self.TPR_score_train[:,j]) for j in range(self.TPR_score_train.shape[1])])
        TPR_test_mean = self.TPR_score_test.mean(axis =0)#np.array([np.mean(self.TPR_score_test[:,j]) for j in range(self.TPR_score_test.shape[1])])
        TPR_test_sd = self.TPR_score_test.std(axis = 0)#np.array([np.std(self.TPR_score_test[:,j]) for j in range(self.TPR_score_test.shape[1])])
        FPR_train_mean = self.FPR_score_train.mean(axis = 0)#np.array([np.mean(self.FPR_score_train[:,j]) for j in range(self.FPR_score_train.shape[1])])
        #FPR_train_sd = np.array([np.std(self.FPR_score_train[:,j]) for j in range(self.FPR_score_train.shape[1])])
        FPR_test_mean = self.FPR_score_test.mean(axis = 0)#np.array([np.mean(self.FPR_score_test[:,j]) for j in range(self.FPR_score_test.shape[1])])
        #FPR_test_sd = np.array([np.std(self.FPR_score_test[:,j]) for j in range(self.FPR_score_test.shape[1])])
        mean_train_auc = np.abs(np.trapz(TPR_train_mean,FPR_train_mean))
        mean_test_auc = np.abs(np.trapz(TPR_test_mean,FPR_test_mean))
        # ix_train = np.argmin(np.sqrt(TPR_train_mean**2-(1-FPR_train_mean)**2))
        # ix_test = np.argmin(np.sqrt(TPR_test_mean**2-(1-FPR_test_mean)**2))

        _,ax = plt.subplots(nrows = 2,ncols = 1,figsize = (15,15))
        ax[0].set_title(f"Mean ROC curve from {self.k} fold CV training set")
        ax[0].set_xlabel("False Positive Rate")
        ax[0].set_ylabel("True Positive Rate")
        ax[0].grid()
        ax[0].scatter(FPR_train_mean[self.ix_tr], TPR_train_mean[self.ix_tr], marker='o', color='black', label='Best')
        ax[0].errorbar(FPR_train_mean,TPR_train_mean,yerr = TPR_train_sd,label = f"{self.name_f} AUC = {mean_train_auc}")
        ax[0].legend(loc = 4)
        ax[1].set_title(f"Mean ROC curve from {self.k} fold CV testing set")
        ax[1].set_xlabel("False Positive Rate")
        ax[1].set_ylabel("True Positive Rate")
        ax[1].scatter(FPR_test_mean[self.ix_t], TPR_test_mean[self.ix_t], marker='o', color='black', label='Best')
        ax[1].errorbar(FPR_test_mean,TPR_test_mean,yerr = TPR_test_sd,label = f"{self.name_f} AUC = {mean_test_auc}")
        ax[1].grid()
        ax[1].legend(loc = 4)
        # print("From training ROC curve : T_optimal for {} = {}".format(self.name_f,self.T[ix_train]))
        # print("From Test ROC curve : T_optimal for {} = {}".format(self.name_f,self.T[ix_test]))


    def plot_PR_curve(self):
        PREC_train_mean = self.Prec_score_train.mean(axis=0)#np.array([np.mean(self.Prec_score_train[:,j]) for j in range(self.Prec_score_train.shape[1])])
        PREC_train_sd = self.Prec_score_train.std(axis=0)#np.array([np.std(self.Prec_score_train[:,j]) for j in range(self.Prec_score_train.shape[1])])
        PREC_test_mean = self.Prec_score_train.mean(axis=0)#np.array([np.mean(self.Prec_score_test[:,j]) for j in range(self.Prec_score_test.shape[1])])
        PREC_test_sd = self.Prec_score_test.std(axis=0)#np.array([np.std(self.Prec_score_test[:,j]) for j in range(self.Prec_score_test.shape[1])])
        REC_train_mean = self.Recall_score_train.mean(axis=0)#np.array([np.mean(self.Recall_score_train[:,j]) for j in range(self.Recall_score_train.shape[1])])
        REC_train_sd = self.Recall_score_train.std(axis=0)#np.array([np.std(self.Recall_score_train[:,j]) for j in range(self.Recall_score_train.shape[1])])
        REC_test_mean = self.Recall_score_test.mean(axis=0)#np.array([np.mean(self.Recall_score_test[:,j]) for j in range(self.Recall_score_test.shape[1])])
        REC_test_sd = self.Recall_score_test.std(axis=0)#np.array([np.std(self.Recall_score_test[:,j]) for j in range(self.Recall_score_test.shape[1])])
        mean_train_auc = np.abs(np.trapz(PREC_train_mean,REC_train_mean))
        mean_test_auc = np.abs(np.trapz(PREC_test_mean,REC_test_mean))
        ix_train = np.argmin(np.sqrt((1-PREC_train_mean)**2+(1-REC_train_mean)**2))
        ix_test = np.argmin(np.sqrt((1-PREC_test_mean)**2+(1-REC_test_mean)**2))
        _,ax = plt.subplots(nrows = 2,ncols = 1,figsize = (15,15))
        ax[0].set_title(f"Mean PR curve from {self.k} fold CV training set")
        ax[0].set_xlabel("Recall")
        ax[0].set_ylabel("Precision")
        ax[0].scatter(REC_train_mean[ix_train], PREC_train_mean[ix_train], marker='o', color='black', label='Best')
        ax[0].errorbar(REC_train_mean,PREC_train_mean,yerr = PREC_train_sd,label = f"{self.name_f} AUC = {mean_train_auc}")
        ax[0].grid()
        ax[0].legend(loc = 4)
        ax[1].set_title(f"Mean PR curve from {self.k} fold CV testing set")
        ax[1].set_xlabel("Recall")
        ax[1].set_ylabel("Precision")
        ax[1].scatter(REC_test_mean[ix_test], PREC_test_mean[ix_test], marker='o', color='black', label='Best')
        ax[1].errorbar(REC_test_mean,PREC_test_mean,yerr = PREC_test_sd, label =f"{self.name_f} AUC = {mean_test_auc}")
        ax[1].grid()
        ax[1].legend(loc = 4)
        print("From training PR curve : T_optimal = ",self.T[ix_train])
        print("From Test PR curve : T_optimal = ",self.T[ix_test])

        ##Return optimal precision recall for train and testing :
        opt_mean_train_prec = PREC_train_mean[self.ix_tr]
        opt_SD_train_prec = PREC_train_sd[self.ix_tr]
        opt_mean_train_recall = REC_train_mean[self.ix_tr]
        opt_SD_train_recall = REC_train_sd[self.ix_tr]
        opt_mean_test_prec = PREC_test_mean[self.ix_t]
        opt_SD_test_prec = PREC_test_sd[self.ix_t]
        opt_mean_test_recall = REC_test_mean[self.ix_t]
        opt_SD_test_recall = REC_test_sd[self.ix_t]

        ##Print them in good format :

        print(f"For Training, at F1 optimal threshold : Precision = {opt_mean_train_prec} +- {opt_SD_train_prec} ; Recall= {opt_mean_train_recall} +- {opt_SD_train_recall}")
        print(f"For Testing, at F1 optimal threshold : Precision = {opt_mean_test_prec} +- {opt_SD_test_prec} ; Recall= {opt_mean_test_recall} +- {opt_SD_test_recall}")

    def Accuracy_calculator(self):

        ACC_train_mean = self.Acc_score_train.mean(axis=0)#np.array([np.mean(self.Acc_score_train[:,j]) for j in range(self.Acc_score_train.shape[1])])
        ACC_train_sd = self.Acc_score_train.std(axis=0)#np.array([np.std(self.Acc_score_train[:,j]) for j in range(self.Acc_score_train.shape[1])])
        ACC_test_mean = self.Acc_score_test.mean(axis=0)#np.array([np.mean(self.Acc_score_test[:,j]) for j in range(self.Acc_score_test.shape[1])])
        ACC_test_sd = self.Acc_score_train.std(axis=0)#np.array([np.std(self.Acc_score_test[:,j]) for j in range(self.Acc_score_test.shape[1])])

        opt_mean_train_acc = ACC_train_mean[self.ix_tr]
        opt_SD_train_acc = ACC_train_sd[self.ix_tr]
        opt_mean_test_acc = ACC_test_mean[self.ix_t]
        opt_SD_test_acc = ACC_test_sd[self.ix_t]

        print(f"For Training, at F1 optimal threshold : Accuracy = {opt_mean_train_acc} +- {opt_SD_train_acc}")
        print(f"For Testing, at F1 optimal threshold :  Accuracy = {opt_mean_test_acc} +- {opt_SD_test_acc}")

    def _get_params(self):
        PREC_train_mean = self.Prec_score_train.mean(axis=0)#np.array([np.mean(self.Prec_score_train[:,j]) for j in range(self.Prec_score_train.shape[1])])
        PREC_test_mean = self.Prec_score_test.mean(axis = 0)#np.array([np.mean(self.Prec_score_test[:,j]) for j in range(self.Prec_score_test.shape[1])])
        PREC_train_sd = self.Prec_score_train.std(axis = 0)#np.array([np.std(self.Prec_score_train[:,j]) for j in range(self.Prec_score_train.shape[1])])
        PREC_test_sd = self.Prec_score_test.std(axis=0)#np.array([np.std(self.Prec_score_test[:,j]) for j in range(self.Prec_score_test.shape[1])])
        REC_train_mean = self.Recall_score_train.mean(axis=0)#np.array([np.mean(self.Recall_score_train[:,j]) for j in range(self.Recall_score_train.shape[1])])
        REC_test_mean = self.Recall_score_test.mean(axis = 0)#np.array([np.mean(self.Recall_score_test[:,j]) for j in range(self.Recall_score_test.shape[1])])
        TPR_train_mean = self.TPR_score_train.mean(axis=0)#np.array([np.mean(self.TPR_score_train[:,j]) for j in range(self.TPR_score_train.shape[1])])
        TPR_train_sd = self.TPR_score_train.std(axis=0)#np.array([np.std(self.TPR_score_train[:,j]) for j in range(self.TPR_score_train.shape[1])])
        TPR_test_sd = self.TPR_score_test.std(axis=0)#np.array([np.std(self.TPR_score_test[:,j]) for j in range(self.TPR_score_test.shape[1])])
        TPR_test_mean = self.TPR_score_test.mean(axis=0)#np.array([np.mean(self.TPR_score_test[:,j]) for j in range(self.TPR_score_test.shape[1])])
        FPR_train_mean = self.FPR_score_train.mean(axis=0)#np.array([np.mean(self.FPR_score_train[:,j]) for j in range(self.FPR_score_train.shape[1])])
        FPR_test_mean = self.FPR_score_test.mean(axis=0)#np.array([np.mean(self.FPR_score_test[:,j]) for j in range(self.FPR_score_test.shape[1])])

        dic_param = {"PR curve Training" : [PREC_train_mean,REC_train_mean,PREC_train_sd] ,
        "PR curve Testing" : [PREC_test_mean,REC_test_mean,PREC_test_sd], "ROC curve Training" : [FPR_train_mean,TPR_train_mean,TPR_train_sd],"ROC curve Testing" : [FPR_test_mean,TPR_test_mean,TPR_test_sd]}
        return dic_param

    def print_prediction_model(self,index,Toptimal,interval):

        if self.name_f == "SQA no TSD":
            X_data = self.Data[index,:,:].T
            y = self.y.copy()
            y_patient = y[index]
            Our_SQA_method.SQANTSD_wrong_estimate(X_data,self.fs,self.ECG_lead,y_patient,Toptimal,interval)
        elif self.name_f =="SQA" :
            X_data = self.Data[index,:,:].T
            y = self.y.copy()
            y_patient = y[index]
            Our_SQA_method.SQA_wrong_estimate(X_data,self.fs,self.ECG_lead,y_patient,Toptimal,interval)
        else:
            raise ValueError("This function can only be for SQA method")
