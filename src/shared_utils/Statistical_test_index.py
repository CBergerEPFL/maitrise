from petastorm import make_reader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.widgets import TextBox, Button
import sys
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score,auc,roc_curve,precision_recall_curve,matthews_corrcoef,precision_recall_fscore_support,confusion_matrix,roc_auc_score
from sklearn.model_selection import StratifiedKFold
import xarray as xr
sys.path.append(os.path.join(os.getcwd(), ".."))
from Metrics import Our_SQA_method
import shared_utils.utils_data as utils_data


save_path = "/workspaces/maitrise/results"
class Statistic_reader():
    def __init__(self,path_peta,function,name_function,Threshold,cv_k = 10,**kwargs):

        if kwargs.get("opposite"):
            self.alternate = kwargs["opposite"]
        else :
            self.alternate= False
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
        with make_reader(path_peta) as reader:
            for sample in reader:
                self.ECG_lead = sample.signal_names
                self.fs = sample.sampling_frequency
                break

        if not "ecg_data.nc" in os.listdir(save_path):
            ds_data = utils_data.format_data_to_xarray(path_peta, save_path)
        else:
            ds_data = xr.load_dataset(os.path.join(save_path,"ecg_data.nc"))

        ds_filtered = ds_data.where(ds_data.data_quality != "unlabeled").dropna(dim = "id")
        self.Data = ds_filtered.signal.values
        self.names = ds_filtered.id.values.astype(int)
        self.y = ds_filtered.data_quality.values
        self.X_data = Statistic_reader.create_dataset(self)


        self.y[self.y=="acceptable"] = 1
        self.y[self.y=="unacceptable"] = 0
        self.y = self.y.astype("int")
        self.y_opp = 1-self.y.copy()
        self.y_opp = self.y_opp.astype("int")

        self.T = np.linspace(Threshold[0],Threshold[1],500)
        self.F_score_train = np.empty((2,self.k,len(self.T)))
        self.F_score_test = np.empty((2,self.k,len(self.T)))
        self.MCC_score_train = np.empty((2,self.k,len(self.T)))
        self.MCC_score_test = np.empty((2,self.k,len(self.T)))
        self.Acc_score_train = np.empty((2,self.k,len(self.T)))
        self.Acc_score_test = np.empty((2,self.k,len(self.T)))
        self.Prec_score_train = np.empty((2,self.k,len(self.T)))
        self.Prec_score_test =np.empty((2,self.k,len(self.T)))
        self.Recall_score_train = np.empty((2,self.k,len(self.T)))
        self.Recall_score_test = np.empty((2,self.k,len(self.T)))
        self.FPR_score_train = np.empty((2,self.k,len(self.T)))
        self.FPR_score_test = np.empty((2,self.k,len(self.T)))
        self.TPR_score_train = np.empty((2,self.k,len(self.T)))
        self.TPR_score_test = np.empty((2,self.k,len(self.T)))
        self.Specificity_score_train = np.empty((2,self.k,len(self.T)))
        self.Specificity_score_test = np.empty((2,self.k,len(self.T)))
        self.auc_train_ROC = np.empty([2,self.k])
        self.auc_test_ROC = np.empty([2,self.k])
        self.auc_train_PR = np.empty([2,self.k])
        self.auc_test_PR = np.empty([2,self.k])
        self.T_opt_train = np.empty([2,self.k])
        self.T_opt_test = np.empty([2,self.k])
        self.ix_tr = 0
        self.ix_t = 0

    def to_labels(self,pos_probs, threshold,opp = False):

        if self.norma or ((np.min(self.X_data)>=0 and np.max(self.X_data)<=1)):
            if opp == False:
                return (pos_probs > threshold).astype(int)
            else :
                return ((1-pos_probs) > threshold).astype(int)

        else :
            if opp == False:
                return (pos_probs > threshold).astype('int')
            else :
                return (pos_probs < threshold).astype('int')

    def create_dataset(self):
        X_data = np.array([])
        if self.norma:
            for x in range(self.Data.shape[0]):
                val = self.function(self.Data[x,:,:].T,self.fs,normalization = True)
                if self.eval == "minimum":
                    X_data = np.append(X_data,np.min(val))
                elif self.eval == "maximum":
                    X_data = np.append(X_data,np.max(val))
                else :
                    X_data = np.append(X_data,np.mean(val))
        else :
            for x in range(self.Data.shape[0]):
                val = self.function(self.Data[x,:,:].T,self.fs,opposite = self.alternate)
                if self.eval == "minimum":
                    X_data = np.append(X_data,np.min(val))
                elif self.eval == "maximum":
                    X_data = np.append(X_data,np.max(val))
                else :
                    X_data = np.append(X_data,np.mean(val))
        return X_data


    def roc_pr_curve(self,y_true, y_prob,opp = 0):
        fpr = []
        tpr = []
        prec = []
        rec = []
        specificity = []
        for threshold in self.T:
            y_pred = Statistic_reader.to_labels(self,y_prob,threshold,opp)

            fp = np.sum((y_pred == 1) & (y_true == 0))
            tp = np.sum((y_pred == 1) & (y_true == 1))

            fn = np.sum((y_pred == 0) & (y_true == 1))
            tn = np.sum((y_pred == 0) & (y_true == 0))

            if tp ==0 and fp == 0:
                prec.append(1)
                rec.append(0)
            elif fp == 0:
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
            specificity.append(tn/(tn+fp))

        return fpr,tpr,prec,rec,specificity


    def confus_mat_T(self,y_true,y_pred):
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tp = np.sum((y_pred == 1) & (y_true == 1))

        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        if tp ==0 and fp == 0:
            prec = 1
            rec = 0
        elif fp == 0:
            prec = 1
            rec  = tp / (tp + fn)
        elif tp == 0:
            prec = 0
            rec = 0
        else :
            prec = tp / (fp + tp)
            rec = tp / (tp + fn)

        fpr = fp / (fp + tn)
        tpr = tp / (tp + fn)
        specificity = tn/(tn+fp)
        return fpr,tpr,rec,prec,specificity

    def CrossValidation_index_opt_thresh(self):
        cv = StratifiedKFold(n_splits=self.k, random_state=1, shuffle=True)
        y_sel = [self.y,self.y_opp]
        for index,y1 in enumerate(y_sel):
            ind = 0
            for train_index, test_index in cv.split(self.X_data,y1):
                X_train,X_test = self.X_data[train_index].copy(),self.X_data[test_index].copy()
                y_train,y_test = y1[train_index].copy(),y1[test_index].copy()

                F_train = [f1_score(y_train, Statistic_reader.to_labels(self,X_train, t,opp = index)) for t in self.T]
                F_test = [f1_score(y_test, Statistic_reader.to_labels(self,X_test, t,opp = index)) for t in self.T]
                Acc_train = [accuracy_score(y_train,Statistic_reader.to_labels(self,X_train,t,opp=index )) for t in self.T]
                Acc_test = [accuracy_score(y_test,Statistic_reader.to_labels(self,X_test,t,opp=index)) for t in self.T]
                MCC_train = [matthews_corrcoef(y_train, Statistic_reader.to_labels(self,X_train, t,opp=index)) for t in self.T]
                MCC_test = [matthews_corrcoef(y_test, Statistic_reader.to_labels(self,X_test, t,opp=index)) for t in self.T]
                fpr_train,tpr_train,prec_train,rec_train,spec_train = Statistic_reader.roc_pr_curve(self,y_train,X_train,opp=index)
                fpr_test,tpr_test,prec_test,rec_test,spec_test = Statistic_reader.roc_pr_curve(self,y_test,X_test,opp=index)
                self.F_score_train[index,ind,:] = F_train
                self.F_score_test[index,ind,:] = F_test
                self.MCC_score_train[index,ind,:] = MCC_train
                self.MCC_score_test[index,ind,:] = MCC_test
                self.TPR_score_train[index,ind,:] = tpr_train
                self.TPR_score_test[index,ind,:] = tpr_test
                self.FPR_score_train[index,ind,:] = fpr_train
                self.FPR_score_test[index,ind,:] = fpr_test
                self.Acc_score_train[index,ind,:] = Acc_train
                self.Acc_score_test[index,ind,:] = Acc_test
                self.Prec_score_train[index,ind,:]= prec_train
                self.Prec_score_test[index,ind,:]= prec_test
                self.Recall_score_train[index,ind,:]= rec_train
                self.Recall_score_test[index,ind,:]= rec_test
                self.Specificity_score_train[index,ind,:] = spec_train
                self.Specificity_score_test[index,ind,:] = spec_test
                self.T_opt_train[index,ind] = self.T[np.argmax(F_train)]
                self.T_opt_test[index,ind] = self.T[np.argmax(F_test)]
                ind +=1
            index +=1


    def Optimal_threshold_calculator(self):

        if self.alternate :
            index = 1
        else :
            index = 0
        F1_train_mean = self.F_score_train[index,:,:].mean(axis=0)#np.array([np.mean(self.F_score_train[:,j]) for j in range(self.F_score_train.shape[1])])
        F1_train_sd = self.F_score_train[index,:,:].std(axis=0)#np.array([np.std(self.F_score_train[:,j]) for j in range(self.F_score_train.shape[1])])
        F1_test_mean = self.F_score_test[index,:,:].mean(axis=0)#np.array([np.mean(self.F_score_test[:,j]) for j in range(self.F_score_test.shape[1])])
        F1_test_sd = self.F_score_test[index,:,:].std(axis=0)#np.array([np.std(self.F_score_test[:,j]) for j in range(self.F_score_test.shape[1])])
        MCC_train_mean = self.MCC_score_train[index,:,:].mean(axis = 0)
        MCC_test_mean = self.MCC_score_test[index,:,:].mean(axis = 0)
        MCC_train_sd = self.MCC_score_train[index,:,:].std(axis = 0)
        MCC_test_sd = self.MCC_score_test[index,:,:].std(axis = 0)
        ix_train = np.argmax(F1_train_mean)
        ix_test = np.argmax(F1_test_mean)
        index_MCC_train = np.argmax(MCC_train_mean)
        index_MCC_test = np.argmax(MCC_test_mean)
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
        print("Best Threshold for {} dataset of {} : ".format("Training",self.name_f), self.T[index_MCC_train], "with MCC-score : {} +- {}".format(np.max(MCC_train_mean),MCC_train_sd[np.argmax(MCC_train_mean)]))
        print("Best Threshold for {} dataset of {} : ".format("Testing",self.name_f), self.T[index_MCC_test], "with MCC-score : {} +- {}".format(np.max(MCC_test_mean),MCC_test_sd[np.argmax(MCC_test_mean)]))
        print("From Training F1 curve : T_optimal = ",self.T[ix_train])
        print("From Testing F1 curve : T_optimal = ",self.T[ix_test])
        print("From Training MCC curve : T_optimal = ",self.T[index_MCC_train])
        print("From Testing MCC curve : T_optimal = ",self.T[index_MCC_test])
        self.ix_tr = ix_train
        self.ix_t = ix_test
        self.mcc_ix_tr = index_MCC_train
        self.mcc_ix_t = index_MCC_test

    def ROC_index(self):

        if self.alternate :
            index = 1
        else :
            index = 0
        TPR_train_mean = self.TPR_score_train[index,:,:].mean(axis = 0)
        TPR_train_sd = self.TPR_score_train[index,:,:].std(axis = 0)
        TPR_test_mean = self.TPR_score_test[index,:,:].mean(axis =0)#np.array([np.mean(self.TPR_score_test[:,j]) for j in range(self.TPR_score_test.shape[1])])
        TPR_test_sd = self.TPR_score_test[index,:,:].std(axis = 0)#np.array([np.std(self.TPR_score_test[:,j]) for j in range(self.TPR_score_test.shape[1])])
        FPR_train_mean = self.FPR_score_train[index,:,:].mean(axis = 0)#np.array([np.mean(self.FPR_score_train[:,j]) for j in range(self.FPR_score_train.shape[1])])
        FPR_train_sd = self.FPR_score_train[index,:,:].std(axis=0)#np.array([np.std(self.FPR_score_train[:,j]) for j in range(self.FPR_score_train.shape[1])])
        FPR_test_mean = self.FPR_score_test[index,:,:].mean(axis = 0)#np.array([np.mean(self.FPR_score_test[:,j]) for j in range(self.FPR_score_test.shape[1])])
        FPR_test_sd = self.FPR_score_test[index,:,:].std(axis=0)
        Specificity_train_mean = self.Specificity_score_train[index,:,:].mean(axis = 0)
        Specificity_train_std = self.Specificity_score_train[index,:,:].std(axis = 0)
        Specificity_test_mean = self.Specificity_score_test[index,:,:].mean(axis = 0)
        Specificity_test_std = self.Specificity_score_test[index,:,:].std(axis = 0)
        ix_train = np.argmin(np.sqrt(TPR_train_mean**2-(1-FPR_train_mean)**2))
        ix_test = np.argmin(np.sqrt(TPR_test_mean**2-(1-FPR_test_mean)**2))
        print("From training ROC curve : T_optimal for {} = {}".format(self.name_f,self.T[ix_train]))
        print("From Test ROC curve : T_optimal for {} = {}".format(self.name_f,self.T[ix_test]))
        print(f"For Training, at F1 optimal threshold : Sensitivity (TPR) = {TPR_train_mean[self.ix_tr]} +- {TPR_train_sd[self.ix_tr]} ; Specificity (TNR)= {Specificity_train_mean[self.ix_tr]} +- {Specificity_train_std[self.ix_tr]}")
        print(f"For Testing, at F1 optimal threshold : Sensitivity (TPR) = {TPR_test_mean[self.ix_t]} +- {TPR_test_sd[self.ix_t]} ; Specificity (TNR)= {Specificity_test_mean[self.ix_t]} +- {Specificity_test_std[self.ix_t]}")
        print(f"For Training, at MCC optimal threshold : Sensitivity (TPR) = {TPR_train_mean[self.mcc_ix_tr]} +- {TPR_train_sd[self.mcc_ix_tr]} ; Specificity (TNR)= {Specificity_train_mean[self.mcc_ix_tr]} +- {Specificity_train_std[self.mcc_ix_tr]}")
        print(f"For Testing, at MCC optimal threshold : Sensitivity (TPR) = {TPR_test_mean[self.mcc_ix_t]} +- {TPR_test_sd[self.mcc_ix_t]} ; Specificity (TNR)= {Specificity_test_mean[self.mcc_ix_t]} +- {Specificity_test_std[self.mcc_ix_t]}")


    def PR_index(self):
        if self.alternate :
            index = 1
        else :
            index = 0
        PREC_train_mean = self.Prec_score_train[index,:,:].mean(axis=0)#np.array([np.mean(self.Prec_score_train[:,j]) for j in range(self.Prec_score_train.shape[1])])
        PREC_train_sd = self.Prec_score_train[index,:,:].std(axis=0)#np.array([np.std(self.Prec_score_train[:,j]) for j in range(self.Prec_score_train.shape[1])])
        PREC_test_mean = self.Prec_score_train[index,:,:].mean(axis=0)#np.array([np.mean(self.Prec_score_test[:,j]) for j in range(self.Prec_score_test.shape[1])])
        PREC_test_sd = self.Prec_score_test[index,:,:].std(axis=0)#np.array([np.std(self.Prec_score_test[:,j]) for j in range(self.Prec_score_test.shape[1])])
        REC_train_mean = self.Recall_score_train[index,:,:].mean(axis=0)#np.array([np.mean(self.Recall_score_train[:,j]) for j in range(self.Recall_score_train.shape[1])])
        REC_train_sd = self.Recall_score_train[index,:,:].std(axis=0)#np.array([np.std(self.Recall_score_train[:,j]) for j in range(self.Recall_score_train.shape[1])])
        REC_test_mean = self.Recall_score_test[index,:,:].mean(axis=0)#np.array([np.mean(self.Recall_score_test[:,j]) for j in range(self.Recall_score_test.shape[1])])
        REC_test_sd = self.Recall_score_test[index,:,:].std(axis=0)#np.array([np.std(self.Recall_score_test[:,j]) for j in range(self.Recall_score_test.shape[1])])
        ix_train = np.argmin(np.sqrt((1-PREC_train_mean)**2+(1-REC_train_mean)**2))
        ix_test = np.argmin(np.sqrt((1-PREC_test_mean)**2+(1-REC_test_mean)**2))
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

        ##Using MCC optimal threshold
        opt_mean_train_prec = PREC_train_mean[self.mcc_ix_tr]
        opt_SD_train_prec = PREC_train_sd[self.mcc_ix_tr]
        opt_mean_train_recall = REC_train_mean[self.mcc_ix_tr]
        opt_SD_train_recall = REC_train_sd[self.mcc_ix_tr]
        opt_mean_test_prec = PREC_test_mean[self.mcc_ix_t]
        opt_SD_test_prec = PREC_test_sd[self.mcc_ix_t]
        opt_mean_test_recall = REC_test_mean[self.mcc_ix_t]
        opt_SD_test_recall = REC_test_sd[self.mcc_ix_t]

        print(f"For Training, at MCC optimal threshold : Precision = {opt_mean_train_prec} +- {opt_SD_train_prec} ; Recall= {opt_mean_train_recall} +- {opt_SD_train_recall}")
        print(f"For Testing, at MCC optimal threshold : Precision = {opt_mean_test_prec} +- {opt_SD_test_prec} ; Recall= {opt_mean_test_recall} +- {opt_SD_test_recall}")

    def Accuracy_calculator(self):
        if self.alternate:
            index = 1
        else :
            index = 0

        ACC_train_mean = self.Acc_score_train[index,:,:].mean(axis=0)#np.array([np.mean(self.Acc_score_train[:,j]) for j in range(self.Acc_score_train.shape[1])])
        ACC_train_sd = self.Acc_score_train[index,:,:].std(axis=0)#np.array([np.std(self.Acc_score_train[:,j]) for j in range(self.Acc_score_train.shape[1])])
        ACC_test_mean = self.Acc_score_test[index,:,:].mean(axis=0)#np.array([np.mean(self.Acc_score_test[:,j]) for j in range(self.Acc_score_test.shape[1])])
        ACC_test_sd = self.Acc_score_train[index,:,:].std(axis=0)#np.array([np.std(self.Acc_score_test[:,j]) for j in range(self.Acc_score_test.shape[1])])

        opt_mean_train_acc = ACC_train_mean[self.ix_tr]
        opt_SD_train_acc = ACC_train_sd[self.ix_tr]
        opt_mean_test_acc = ACC_test_mean[self.ix_t]
        opt_SD_test_acc = ACC_test_sd[self.ix_t]

        print(f"For Training, at F1 optimal threshold : Accuracy = {opt_mean_train_acc} +- {opt_SD_train_acc}")
        print(f"For Testing, at F1 optimal threshold :  Accuracy = {opt_mean_test_acc} +- {opt_SD_test_acc}")

        opt_mean_train_acc = ACC_train_mean[self.mcc_ix_tr]
        opt_SD_train_acc = ACC_train_sd[self.mcc_ix_tr]
        opt_mean_test_acc = ACC_test_mean[self.mcc_ix_t]
        opt_SD_test_acc = ACC_test_sd[self.mcc_ix_t]

        print(f"For Training, at MCC optimal threshold : Accuracy = {opt_mean_train_acc} +- {opt_SD_train_acc}")
        print(f"For Testing, at MCC optimal threshold :  Accuracy = {opt_mean_test_acc} +- {opt_SD_test_acc}")

    def _get_params(self):
        if self.alternate :
            index = 1
        else :
            index = 0
        PREC_train_mean = self.Prec_score_train[index,:,:].mean(axis=0)#np.array([np.mean(self.Prec_score_train[:,j]) for j in range(self.Prec_score_train.shape[1])])
        PREC_test_mean = self.Prec_score_test[index,:,:].mean(axis = 0)#np.array([np.mean(self.Prec_score_test[:,j]) for j in range(self.Prec_score_test.shape[1])])
        PREC_train_sd = self.Prec_score_train[index,:,:].std(axis = 0)#np.array([np.std(self.Prec_score_train[:,j]) for j in range(self.Prec_score_train.shape[1])])
        PREC_test_sd = self.Prec_score_test[index,:,:].std(axis=0)#np.array([np.std(self.Prec_score_test[:,j]) for j in range(self.Prec_score_test.shape[1])])
        REC_train_mean = self.Recall_score_train[index,:,:].mean(axis=0)#np.array([np.mean(self.Recall_score_train[:,j]) for j in range(self.Recall_score_train.shape[1])])
        REC_test_mean = self.Recall_score_test[index,:,:].mean(axis = 0)#np.array([np.mean(self.Recall_score_test[:,j]) for j in range(self.Recall_score_test.shape[1])])
        TPR_train_mean = self.TPR_score_train[index,:,:].mean(axis=0)#np.array([np.mean(self.TPR_score_train[:,j]) for j in range(self.TPR_score_train.shape[1])])
        TPR_train_sd = self.TPR_score_train[index,:,:].std(axis=0)#np.array([np.std(self.TPR_score_train[:,j]) for j in range(self.TPR_score_train.shape[1])])
        TPR_test_sd = self.TPR_score_test[index,:,:].std(axis=0)#np.array([np.std(self.TPR_score_test[:,j]) for j in range(self.TPR_score_test.shape[1])])
        TPR_test_mean = self.TPR_score_test[index,:,:].mean(axis=0)#np.array([np.mean(self.TPR_score_test[:,j]) for j in range(self.TPR_score_test.shape[1])])
        FPR_train_mean = self.FPR_score_train[index,:,:].mean(axis=0)#np.array([np.mean(self.FPR_score_train[:,j]) for j in range(self.FPR_score_train.shape[1])])
        FPR_test_mean = self.FPR_score_test[index,:,:].mean(axis=0)#np.array([np.mean(self.FPR_score_test[:,j]) for j in range(self.FPR_score_test.shape[1])])

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


    def Plot_PR_fold_graph_homemade(self):
        if self.alternate :
            index = 1
        else :
            index = 0
        aucs = np.array([])
        plt.figure()
        color = iter(plt.cm.rainbow(np.linspace(0, 1, self.k)))
        for i in range(self.k):
            c = next(color)
            plt.plot(self.Recall_score_test[index,i,:],self.Prec_score_test[index,i,:],color=c,label="PR at fold {} with AUC = {:.2f}".format(i,np.abs(np.trapz(self.Prec_score_test[index,i,:],self.Recall_score_test[index,i,:]))),alpha=0.3,lw=1)
            aucs = np.append(aucs,np.abs(np.trapz(self.Prec_score_test[index,i,:],self.Recall_score_test[index,i,:])))
        plt.plot([0, 1], [0, 0], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
        mean_tpr = np.mean(self.Prec_score_test[index,:,:], axis=0)
        mean_fpr = np.mean(self.Recall_score_test[index,:,:], axis=0)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        plt.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean PR (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(self.Prec_score_test[index,:,:], axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )
        plt.legend()
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR curve of each fold for index {self.name_f}")
        plt.grid()

    def Plot_ROC_fold_graph_homemade(self):
        if self.alternate :
            index = 1
        else :
            index = 0
        aucs = np.array([])
        plt.figure()
        color = iter(plt.cm.rainbow(np.linspace(0, 1, self.k)))
        for i in range(self.k):
            c = next(color)
            plt.plot(self.FPR_score_test[index,i,:],self.TPR_score_test[index,i,:],color=c,label="ROC at fold {} with AUC = {:.2f}".format(i,np.abs(np.trapz(self.TPR_score_test[index,i,:],self.FPR_score_test[index,i,:]))),alpha=0.3,lw=1)
            aucs = np.append(aucs,np.abs(np.trapz(self.TPR_score_test[index,i,:],self.FPR_score_test[index,i,:])))

        plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
        mean_tpr = np.mean(self.TPR_score_test[index,:,:], axis=0)
        mean_fpr = np.mean(self.FPR_score_test[index,:,:], axis=0)
        mean_auc = np.mean(aucs)#np.abs(np.trapz(self.TPR_score_test[i,:],self.FPR_score_test[i,:]))
        std_auc = np.std(aucs)
        plt.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(self.TPR_score_test[index,:,:], axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )
        plt.legend()
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC curve of each fold for index {self.name_f}")
        plt.grid()

    def ROC_PR_curve_homemade(self):
        if self.alternate :
            index = 1
        else :
            index = 0
        aucs_roc = np.array([])
        aucs_pr = np.array([])
        fig, ax = plt.subplots(nrows = 2,ncols =1,figsize = (15,15))
        color = iter(plt.cm.rainbow(np.linspace(0, 1, self.k)))
        ax[0].plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
        ax[1].plot([0, 1], [0, 0], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
        for i in range(self.k):
            c = next(color)
            ax[0].plot(self.FPR_score_test[index,i,:],self.TPR_score_test[index,i,:],color=c,label="ROC at fold {} with AUC = {:.2f}".format(i,np.abs(np.trapz(self.TPR_score_test[index,i,:],self.FPR_score_test[index,i,:]))),alpha=0.3,lw=1)
            aucs_roc = np.append(aucs_roc,np.abs(np.trapz(self.TPR_score_test[index,i,:],self.FPR_score_test[index,i,:])))
            ax[1].plot(self.Recall_score_test[index,i,:],self.Prec_score_test[index,i,:],color=c,label="PR at fold {} with AUC = {:.2f}".format(i,np.abs(np.trapz(self.Prec_score_test[index,i,:],self.Recall_score_test[index,i,:]))),alpha=0.3,lw=1)
            aucs_pr = np.append(aucs_pr,np.abs(np.trapz(self.Prec_score_test[index,i,:],self.Recall_score_test[index,i,:])))

        mean_tpr = np.mean(self.TPR_score_test[index,:,:], axis=0)
        mean_fpr = np.mean(self.FPR_score_test[index,:,:], axis=0)
        mean_auc_roc = np.mean(aucs_roc)
        std_auc_roc = np.std(aucs_roc)

        mean_prec = np.mean(self.Prec_score_test[index,:,:], axis=0)
        mean_rec = np.mean(self.Recall_score_test[index,:,:], axis=0)
        mean_auc_pr = np.mean(aucs_pr)
        std_auc_pr = np.std(aucs_pr)

        ax[0].plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc_roc, std_auc_roc),
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(self.TPR_score_test[index,:,:], axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax[0].fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )
        ax[0].legend()
        ax[0].set_xlabel("False Positive Rate")
        ax[0].set_ylabel("True Positive Rate")
        ax[0].set_title(f"ROC curve of each fold for index {self.name_f} from homemade method")
        ax[0].grid()

        ax[1].plot(
            mean_rec,
            mean_prec,
            color="b",
            label=r"Mean PR (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc_pr, std_auc_pr),
            lw=2,
            alpha=0.8,
        )

        std_prec = np.std(self.Prec_score_test[index,:,:], axis=0)
        precs_upper = np.minimum(mean_prec + std_prec, 1)
        precs_lower = np.maximum(mean_prec - std_prec, 0)
        ax[1].fill_between(
            mean_rec,
            precs_lower,
            precs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )
        ax[1].legend()
        ax[1].set_xlabel("Recall")
        ax[1].set_ylabel("Precision")
        ax[1].set_title(f"PR curve of each fold for index {self.name_f} from homemade method")
        ax[1].grid()



    def ROC_PR_curve_sklearn(self):
        cv = StratifiedKFold(n_splits=self.k, random_state=1, shuffle=True)
        mean_fpr = np.linspace(0,1,500)
        mean_recall = np.linspace(0,1,500)
        tprs = []
        precs = []
        aucs_roc = []
        aucs_pr = []
        if self.alternate :
            y = self.y_opp
        else :
            y = self.y

        for _, (train, test) in enumerate(cv.split(self.X_data, y.ravel())):
            if self.norma and self.alternate or ((np.min(self.X_data)>=0 and np.max(self.X_data)<=1) and self.alternate):
                fpr,tpr,_ = roc_curve(y[test],1-self.X_data[test],pos_label = 1)
            elif (self.norma and not self.alternate) or (not self.norma and not self.alternate):
                fpr,tpr,_ = roc_curve(y[test],self.X_data[test],pos_label = 1)
            else :
                raise AttributeError("You cannot inverse the labeling if your index is not normalized! here : min val = ",np.min(self.X_data)," and max val : ",np.max(self.X_data))
            interp_tpr = np.interp(mean_fpr,fpr,tpr)
            interp_tpr[0]= 0
            tprs.append(interp_tpr)
            aucs_roc.append(auc(fpr,tpr))

            if self.norma and self.alternate or ((np.min(self.X_data)>=0 and np.max(self.X_data)<=1) and self.alternate):
                precision,recall,_ = precision_recall_curve(y[test],1-self.X_data[test],pos_label = 1)
            elif (self.norma and not self.alternate) or (not self.norma and not self.alternate):
                precision,recall,_ = precision_recall_curve(y[test],self.X_data[test],pos_label = 1)
            else:
                raise AttributeError("You cannot inverse the labeling if your index is not normalized!")
            index_rec = np.argsort(recall)
            interp_prec = np.interp(mean_recall,np.sort(recall),precision[index_rec])
            precs.append(interp_prec)
            aucs_pr.append(auc(recall,precision))


        precision_avg = np.mean(precs,axis = 0)

        mean_auc_pr = np.mean(aucs_pr)
        std_auc_pr = np.std(aucs_pr)
        std_precs = np.std(precs, axis=0)
        precs_upper = np.minimum(precision_avg + std_precs, 1)
        precs_lower = np.maximum(precision_avg  - std_precs, 0)

        tpr_avg = np.mean(tprs,axis = 0)
        mean_auc_roc = np.mean(aucs_roc)
        std_auc_roc = np.std(aucs_roc)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(tpr_avg + std_tpr, 1)
        tprs_lower = np.maximum(tpr_avg  - std_tpr, 0)

        fig, ax = plt.subplots(nrows = 2,ncols =1,figsize = (15,15))
        color = iter(plt.cm.rainbow(np.linspace(0, 1, self.k)))

        ax[0].plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
        ax[1].plot([0, 1], [0, 0], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

        for j in range(self.k):
            c = next(color)
            ax[0].plot(mean_fpr,tprs[j],label= "ROC fold {} with AUC = {:.2f}".format(j,aucs_roc[j]),color = c,alpha=0.3,lw=1)
            ax[1].plot(mean_recall,precs[j],label= "PR fold {} with AUC = {:.2f}".format(j,aucs_pr[j]),color = c,alpha=0.3,lw=1)

        ax[0].plot(mean_fpr,tpr_avg,label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc_roc, std_auc_roc),color="b")
        ax[0].fill_between(mean_fpr,tprs_lower,tprs_upper,color="grey",alpha=0.2,label=r"$\pm$ 1 std. dev.")
        ax[0].set_xlabel("FPR")
        ax[0].set_ylabel("TPR")
        ax[0].set_title(f"ROC curve for {self.name_f} using sklearn methods")
        ax[0].grid()
        ax[0].legend(loc = "best")
        ax[1].plot(mean_recall,precision_avg,label=r"Mean PR (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc_pr, std_auc_pr),color="b")
        ax[1].fill_between(mean_recall,precs_lower,precs_upper,color="grey",alpha=0.2,label=r"$\pm$ 1 std. dev.")
        ax[1].set_xlabel("Recall")
        ax[1].set_ylabel("Precision")
        ax[1].set_title(f"PR curve for {self.name_f} using sklearn methods")
        ax[1].grid()
        ax[1].legend(loc = "best")

        plt.show()

        return {"ROC":np.vstack((mean_fpr,tpr_avg)),"PR":np.vstack((mean_recall,precision_avg)),"ROC mean AUC":mean_auc_roc,"ROC std AUC":std_auc_roc,"PR mean AUC":mean_auc_pr,"PR std AUC":std_auc_pr}


    def Classification_Report_index(self):
        cv = StratifiedKFold(n_splits=self.k)
        Precision_arr = np.empty([self.k,2])
        recall_arr = np.empty([self.k,2])
        F1_arr = np.empty([self.k,2])
        Specificity_arr = np.empty([self.k,2])
        MCC_arr = np.empty([self.k,2])
        AUCROC_arr = np.empty([self.k,2])
        AUCPR_arr = np.empty([self.k,2])
        Acc_arr = np.empty([self.k,2])
        t_opt_norm = self.T[np.argmax(self.F_score_test[0,:,:].mean(axis = 0))]
        t_opt_opp = self.T[np.argmax(self.F_score_test[1,:,:].mean(axis = 0))]
        nb_occurence = np.empty([self.k,2])
        for i,(train,test) in enumerate(cv.split(self.X_data,self.y.ravel())):
            y_pred_opp = (1-self.X_data[test]>t_opt_opp).astype("int")
            y_pred = (self.X_data[test]>t_opt_norm).astype("int")

            prec,rec,f1,s = precision_recall_fscore_support(self.y[test],y_pred,labels = [0,1],pos_label = 1)
            prec_opp,rec_opp,f1_opp,_ = precision_recall_fscore_support(1-self.y[test],y_pred_opp,labels = [0,1],pos_label = 1)
            acc = accuracy_score(self.y[test],y_pred)
            acc_opp = accuracy_score(1-self.y[test],y_pred_opp)
            Acc_arr[i,0],Acc_arr[i,1] = acc_opp,acc
            nb_occurence[i,0],nb_occurence[i,1] = s[0],s[1]
            Precision_arr[i,0],Precision_arr[i,1] = prec_opp[1],prec[1]
            recall_arr[i,0],recall_arr[i,1] = rec_opp[1],rec[1]
            F1_arr[i,0],F1_arr[i,1] = f1_opp[1],f1[1]
            _,_,_,_,spec_norm = self.confus_mat_T(self.y[test],y_pred)
            _,_,_,_,spec_opp = self.confus_mat_T(1-self.y[test],y_pred_opp)
            Specificity_arr[i,1] = spec_norm
            Specificity_arr[i,0] = spec_opp
            MCC_norm = matthews_corrcoef(self.y[test],y_pred)
            MCC_opp = matthews_corrcoef(1-self.y[test],y_pred_opp)
            MCC_arr[i,0],MCC_arr[i,1] = MCC_opp,MCC_norm
            AUCROC_arr[i,0],AUCROC_arr[i,1] = roc_auc_score(1-self.y[test],1-self.X_data[test]),roc_auc_score(self.y[test],self.X_data[test])
            prec_norm,rec_norm,_ = precision_recall_curve(self.y[test],self.X_data[test],pos_label = 1)
            prec_opp,rec_opp,_ = precision_recall_curve(1-self.y[test],1-self.X_data[test],pos_label = 1)
            AUCPR_arr[i,0],AUCPR_arr[i,1] = auc(rec_opp,prec_opp),auc(rec_norm,prec_norm)


        Acc_mean_opp = Acc_arr[:,0].mean()
        Acc_std_opp = Acc_arr[:,0].std()
        Acc_mean_norm = Acc_arr[:,1].mean()
        Acc_std_norm = Acc_arr[:,1].std()

        Prec_mean_opp = Precision_arr[:,0].mean()
        Prec_std_opp = Precision_arr[:,0].std()
        Prec_mean_norm = Precision_arr[:,1].mean()
        Prec_std_norm = Precision_arr[:,1].std()

        recall_mean_opp = recall_arr[:,0].mean()
        recall_std_opp = recall_arr[:,0].std()
        recall_mean_norm = recall_arr[:,1].mean()
        recall_std_norm = recall_arr[:,1].std()

        F1_mean_opp = F1_arr[:,0].mean()
        F1_std_opp = F1_arr[:,0].std()
        F1_mean_norm = F1_arr[:,1].mean()
        F1_std_norm = F1_arr[:,1].std()

        MCC_mean_opp = MCC_arr[:,0].mean()
        MCC_std_opp = MCC_arr[:,0].std()
        MCC_mean_norm = MCC_arr[:,1].mean()
        MCC_std_norm = MCC_arr[:,1].std()

        AUCROC_mean_opp = AUCROC_arr[:,0].mean()
        AUCROC_std_opp = AUCROC_arr[:,0].std()
        AUCROC_mean_norm = AUCROC_arr[:,1].mean()
        AUCROC_std_norm = AUCROC_arr[:,1].std()

        AUCPR_mean_opp = AUCPR_arr[:,0].mean()
        AUCPR_std_opp = AUCPR_arr[:,0].std()
        AUCPR_mean_norm = AUCPR_arr[:,1].mean()
        AUCPR_std_norm = AUCPR_arr[:,1].std()

        Spec_mean_opp = Specificity_arr[:,0].mean()
        Spec_std_opp = Specificity_arr[:,0].std()
        Spec_mean_norm = Specificity_arr[:,1].mean()
        Spec_std_norm = Specificity_arr[:,1].std()


        df = pd.DataFrame(index = ["0","1"])
        df["Precision (mean,std) (T_optimal)"] = [(np.around(Prec_mean_opp,2),np.around(Prec_std_opp,2)),(np.around(Prec_mean_norm,2),np.around(Prec_std_norm,2))]
        df["Accuracy (mean,std) (T_optimal)"] = [(np.around(Acc_mean_opp,2),np.around(Acc_std_opp,2)),(np.around(Acc_mean_norm,2),np.around(Acc_std_norm,2))]
        df["Recall (mean,std) (T_optimal)"] = [(np.around(recall_mean_opp,2),np.around(recall_std_opp,2)),(np.around(recall_mean_norm,2),np.around(recall_std_norm,2))]
        df["Specificity (mean,std) (T_optimal)"] = [(np.around(Spec_mean_opp,2),np.around(Spec_std_opp,2)),(np.around(Spec_mean_norm,2),np.around(Spec_std_norm,2))]
        df["F1 score (mean,std) (T_optimal)"] = [(np.around(F1_mean_opp,2),np.around(F1_std_opp,2)),(np.around(F1_mean_norm,2),np.around(F1_std_norm,2))]
        df["MCC (mean,std) (T_optimal)"] = [(np.around(MCC_mean_opp,2),np.around(MCC_std_opp,2)),(np.around(MCC_mean_norm,2),np.around(MCC_std_norm,2))]
        df["AUC ROC (mean,std)"] = [(np.around(AUCROC_mean_opp,2),np.around(AUCROC_std_opp,2)),(np.around(AUCROC_mean_norm,2),np.around(AUCROC_std_norm,2))]
        df["AUC PR (mean,std)"] = [(np.around(AUCPR_mean_opp,2),np.around(AUCPR_std_opp,2)),(np.around(AUCPR_mean_norm,2),np.around(AUCPR_std_norm,2))]
        df["Optimal Threhsold (max Mean F1 score)"] = [t_opt_opp,t_opt_norm]
        df["Number of occurence in test set"] = [nb_occurence[:,0].sum().astype(int),nb_occurence[:,1].sum().astype(int)]




        print(f"Classification Report for index {self.name_f}")
        display(df)
        return df
