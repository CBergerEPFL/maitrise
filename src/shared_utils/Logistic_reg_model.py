import os
import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import statsmodels.api as sm
from sklearn.metrics import auc,precision_recall_curve,roc_auc_score,RocCurveDisplay,PrecisionRecallDisplay,precision_score,recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (RepeatedStratifiedKFold, cross_val_score,
                                     train_test_split,StratifiedKFold,KFold)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
sys.path.append(os.path.join(os.getcwd(), ".."))
from shared_utils.Custom_Logit import Logit_binary
seed = 0

def ROC_PR_CV_curve_model(X_data,y_data,cols=None,k_cv=6,model_type = "Logistic",Feature_selection="Backward Model Selection",pos_lab = 1):
    T_range = [0,1]
    if cols is None:
        print("Using : {}".format(Feature_selection))
        cols = Backward_model_selection(X_data,y_data)
        if "HR" in cols:
            Hindex = list(X_data.columns.values).index("HR")
        else :
            Hindex = None
        X = X_data[cols].values
        y = y_data.values
    else :
        if "HR" in cols:
            Hindex = list(X_data.columns.values).index("HR")
        else :
            Hindex = None
        X = X_data[cols].values
        y = y_data.values

    if model_type == "ExtraTreeClassifier":
        model = ExtraTreesClassifier(random_state=seed)
    elif model_type == "RandomTreeClassifier":
        model = RandomForestClassifier(random_state=seed)
    elif model_type == "Logistic" and Hindex is not None:
        model = Logit_binary(Hindex,random_state = seed)
    else :
        model = LogisticRegression(random_state=seed)

    cv = StratifiedKFold(n_splits=k_cv)
    fprs = []
    tprs = []
    precs = []
    recs = []
    aucs_roc = []
    aucs_pr = []
    T = np.linspace(T_range[0]-1,T_range[1]+1,500)
    X = X_data.values
    y = y_data.values
    for _, (train, test) in enumerate(cv.split(X, y.ravel())):
        model.fit(X[train],y[train].ravel())
        y_score = model.predict_proba(X[test])
        fpr,tpr = roc_curve(y[test].ravel(),y_score[:,pos_lab],T)
        precision,recall = pr_curve(y[test].ravel(),y_score[:,pos_lab],T)
        fprs.append(fpr)
        tprs.append(tpr)
        aucs_roc.append(auc(fpr,tpr))
        precs.append(precision)
        recs.append(recall)
        aucs_pr.append(auc(recall,precision))
    precision_avg = np.mean(precs,axis = 0)
    mean_recall = np.mean(recs,axis = 0)

    mean_auc_pr = np.mean(aucs_pr)#auc(mean_fpr, mean_tpr)
    std_auc_pr = np.std(aucs_pr)
    std_precs = np.std(precs, axis=0)
    precs_upper = np.minimum(precision_avg + std_precs, 1)
    precs_lower = np.maximum(precision_avg  - std_precs, 0)

    tpr_avg = np.mean(tprs,axis = 0)
    mean_fpr = np.mean(fprs,axis = 0)
    mean_auc_roc = np.mean(aucs_roc)#auc(mean_fpr, mean_tpr)
    std_auc_roc = np.std(aucs_roc)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(tpr_avg + std_tpr, 1)
    tprs_lower = np.maximum(tpr_avg  - std_tpr, 0)

    fig, ax = plt.subplots(nrows = 2,ncols =1,figsize = (10,25))
    color = iter(plt.cm.rainbow(np.linspace(0, 1, k_cv)))

    ax[0].plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    ax[1].plot([0, 1], [0, 0], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    for j in range(k_cv):
        c = next(color)
        ax[0].plot(mean_fpr,tprs[j],label= "ROC fold {} with AUC = {:.2f}".format(j,aucs_roc[j]),color = c,alpha=0.3,lw=1)
        ax[1].plot(recs[j],precs[j],label= "PR fold {} with AUC = {:.2f}".format(j,aucs_pr[j]),color = c,alpha=0.3,lw=1)

    ax[0].plot(mean_fpr,tpr_avg,label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc_roc, std_auc_roc),color="b")
    ax[0].fill_between(mean_fpr,tprs_lower,tprs_upper,color="grey",alpha=0.2,label=r"$\pm$ 1 std. dev.")
    ax[0].set_xlabel("FPR")
    ax[0].set_ylabel("TPR")
    ax[0].set_title(f"ROC curve for {model_type}")
    ax[0].grid()
    ax[0].legend(loc = "best")
    ax[1].plot(mean_recall,precision_avg,label=r"Mean PR (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc_pr, std_auc_pr),color="b")
    ax[1].fill_between(mean_recall,precs_lower,precs_upper,color="grey",alpha=0.2,label=r"$\pm$ 1 std. dev.")
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].set_title(f"PR curve for {model_type}")
    ax[1].grid()
    ax[1].legend(loc = "best")

    plt.show()

def Backward_model_selection(X,y,threshold_out = 0.001):
    initial_feature_set = list(X.columns.values)
    logit_model=sm.Logit(y.values.ravel(),X)
    result=logit_model.fit()
    sumsum = results_summary_to_dataframe(result)
    list_pval = np.array(sumsum["pvals"].values)
    max_pval = sumsum["pvals"].max()
    while max_pval>=threshold_out:
        idx_maxPval = np.array(initial_feature_set)[list_pval == max_pval]
        initial_feature_set.remove(idx_maxPval)
        logit_mod = sm.Logit(y,X[initial_feature_set])
        result  = logit_mod.fit()
        sumsum = results_summary_to_dataframe(result)
        max_pval = sumsum["pvals"].max()
        list_pval = np.array(sumsum["pvals"].values)
    return initial_feature_set


def Model_classification_score(X,y,cols):
    X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.3, random_state=seed)
    if len(cols) == 0:
        print("All feature in the dataset selected!")
        cols = X_train.columns

    if "HR" in cols:
        index = cols.index("HR")
    else :
        index = None


    os_data_X = X_train[cols].values
    if index is None :
        logreg = LogisticRegression(random_state=seed)
    else :
        logreg = Logit_binary(index,random_state=seed)
    logreg.fit(os_data_X, y_train.ravel())
    x_test = pd.DataFrame(data = X_test[cols].values,columns = cols)
    x_test = x_test.to_numpy()
    y_pred = logreg.predict(x_test)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(x_test, y_test)))
    cm = confusion_matrix(y_test, y_pred)
    sn.heatmap(cm, annot=True, annot_kws={"size": 16},fmt='g')
    print(classification_report(y_test, y_pred))

def evaluate_model(X_data, y_data, repeats):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=repeats, random_state=seed)
    if "HR" in X_data.columns:
        index = list(X_data.columns.values).index("HR")
        model = Logit_binary(HR_index = index,random_state=seed)
        X = X_data.values
        y = y_data.values.ravel()
    else :
        model = LogisticRegression(random_state=seed)
        X = X_data.values
        y = y_data.values.ravel()

    scores = cross_val_score(model, X, y, scoring='f1', cv=cv, n_jobs=-1)
    return scores

def f1_score_CV_estimates(X,y,repeats):
	results = list()
	for r in range(1,repeats):
		# evaluate using a given number of repeats
		scores = evaluate_model(X, y, r)
	# summarize
		print('>%d mean=%.4f se=%.3f' % (r, np.mean(scores), np.std(scores)))
		# store
		results.append(scores)
	plt.boxplot(results, labels=[str(r) for r in range(repeats)], showmeans=True)
	plt.show()


def results_summary_to_dataframe(results):
    '''take the result of an statsmodel results table and transforms it into a dataframe'''
    pvals = results.pvalues
    coeff = results.params
    conf_lower = results.conf_int()[0]
    conf_higher = results.conf_int()[1]

    results_df = pd.DataFrame({"pvals":pvals,
                               "coeff":coeff,
                               "conf_lower":conf_lower,
                               "conf_higher":conf_higher
                                })

    #Reordering...
    results_df = results_df[["coeff","pvals","conf_lower","conf_higher"]]
    return results_df



def pr_curve(y_true, y_prob,T_r):
    prec = []
    rec = []
    for threshold in T_r:
        y_pred = (y_prob>threshold).astype(int)

        fp = np.sum((y_pred == 1) & (y_true == 0))
        tp = np.sum((y_pred == 1) & (y_true == 1))

        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        if fp == 0 and tp==0:
            prec.append(0)
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

    return prec,rec

def ExtraTreeClassifier_CV_Feature_selection(X_data,y_data,k_cv = 10):
    model = ExtraTreesClassifier(random_state=seed)
    cv = StratifiedKFold(n_splits=k_cv)
    cols = X_data.columns.values
    df = pd.DataFrame(index = X_data.columns)
    X = X_data.values
    y = y_data.values
    for i, (train, test) in enumerate(cv.split(X, y.ravel())):
        model.fit(X[train],y[train].ravel())
        feat_importances = pd.Series(model.feature_importances_, index=X_data.columns)
        df[f"{i+1} fold"] = feat_importances
    df_n = df.to_numpy()
    mean_val = np.mean(df_n,axis = 1)
    std_val = np.std(df_n,axis = 1)
    plt.figure()
    plt.bar(cols,mean_val)
    plt.errorbar(cols,mean_val,yerr = std_val,alpha=0.5,fmt = "o",color = "r", ecolor='black', capsize=10)
    plt.title(f"Feature importance from ExtraTreeClassifier for {k_cv} Fold CV on training set")
    plt.xlabel("Features")
    plt.ylabel("Gini Score")
    plt.grid()
    plt.tight_layout()
    plt.show()

def Kbest_MutulaInformation_CV(X_data,y_data,k_cv=10):
    model = SelectKBest(score_func = mutual_info_classif,k = len(X_data.columns.values))
    cv = StratifiedKFold(n_splits=k_cv)
    cols = X_data.columns.values
    df = pd.DataFrame(index = X_data.columns)
    X = X_data.values
    y = y_data.values
    for i, (train, test) in enumerate(cv.split(X, y.ravel())):
        fit = model.fit(X[train],y[train].ravel())
        df[f"{i+1} fold"] = pd.DataFrame(fit.scores_,index=cols)

    df_n = df.to_numpy()
    mean_val = np.mean(df_n,axis = 1)
    std_val = np.std(df_n,axis = 1)
    plt.figure()
    plt.bar(cols,mean_val)
    plt.errorbar(cols,mean_val,yerr = std_val,alpha=0.5,fmt = "o",color = "r", ecolor='black', capsize=10)
    plt.title(f"Mutual information for {k_cv} Fold CV on training set")
    plt.xlabel("Features")
    plt.ylabel("Mutual Information")
    plt.grid()
    plt.tight_layout()
    plt.show()

def roc_curve(y_true, y_prob,T_r):
    fpr = []
    tpr = []
    for threshold in T_r:
        y_pred = (y_prob>threshold).astype(int)

        fp = np.sum((y_pred == 1) & (y_true == 0))
        tp = np.sum((y_pred == 1) & (y_true == 1))

        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))

    return fpr,tpr
