import os
import sys
import warnings
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import statsmodels.api as sm
from sklearn.metrics import auc,roc_curve,precision_recall_curve,roc_auc_score,RocCurveDisplay,PrecisionRecallDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (RepeatedStratifiedKFold, cross_val_score,
                                     train_test_split,StratifiedKFold,KFold)

seed = 0
def ROC_CV_curve(X_data,y_data,k_cv=6,cols = None,Feature_selection = "Backward Model Selection"):
    cv = KFold(n_splits=k_cv)
    classifier = LogisticRegression(random_state=seed)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    if cols == None:
        print("Using : {}".format(Feature_selection))
        cols = Backward_model_selection(X_data,y_data)
        X = X_data[cols].values
        y = y_data.values
    else :
        X = X_data[cols].values
        y = y_data.values


    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X, y.ravel())):
        classifier.fit(X[train], y[train].ravel())
        viz = RocCurveDisplay.from_estimator(
            classifier,
            X[test],
            y[test].ravel(),
            name="ROC fold {}".format(i),
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title="Receiver operating characteristic Logistic Regression model using {}".format(cols),
    )
    ax.legend(loc="lower right")
    plt.show()

def PR_CV_curve(X_data,y_data,k_cv=6,cols = None,Feature_selection = "Backward Model Selection"):
    cv = StratifiedKFold(n_splits=k_cv)
    classifier = LogisticRegression(random_state=seed)

    Precs = []
    aucs = []
    mean_Recs = np.linspace(0, 1, 100)

    if cols == None:
        print("Using : {}".format(Feature_selection))
        cols = Backward_model_selection(X_data,y_data)
        X = X_data[cols].values
        y = y_data.values
    else :
        X = X_data[cols].values
        y = y_data.values

    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X, y.ravel())):
        classifier.fit(X[train], y[train].ravel())
        viz = PrecisionRecallDisplay.from_estimator(
            classifier,
            X[test],
            y[test].ravel(),
            name="ROC fold {}".format(i),
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_prec = np.interp(mean_Recs, viz.recall, viz.precision)
        #interp_prec[0] = 1.0
        Precs.append(interp_prec)
        aucs.append(viz.average_precision)


    ax.plot([0, 1], [0, 0], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_prec = np.mean(Precs, axis=0)
    #mean_prec[-1] = 0.0
    mean_auc = auc(mean_Recs, mean_prec)
    std_auc = np.std(aucs)
    ax.plot(
        mean_Recs,
        mean_prec,
        color="b",
        label=r"Mean PR (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(Precs, axis=0)
    tprs_upper = np.minimum(mean_prec + std_tpr, 1)
    tprs_lower = np.maximum(mean_prec - std_tpr, 0)
    ax.fill_between(
        mean_Recs,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title="Precision Recall Logistic Regression model using {}".format(cols),
    )
    ax.legend(loc="lower right")
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
    columns = X_train.columns
    os_data_X = X_train[cols]
    logreg = LogisticRegression(random_state=seed)
    logreg.fit(os_data_X, y_train)
    x_test = pd.DataFrame(data = X_test,columns = columns)
    x_test = x_test[cols].to_numpy()
    y_pred = logreg.predict(x_test)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(x_test, y_test)))
    cm = confusion_matrix(y_test, y_pred)
    sn.heatmap(cm)
    print(classification_report(y_test, y_pred))

def ROC_curve_model(model,y_test,x_test):
    logit_roc_auc = roc_auc_score(y_test, model.predict(x_test))
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(x_test)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic without using SMOTE')
    plt.legend(loc="lower right")
    plt.show()


def PR_curve_model(model,y_test,x_test):
    precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(x_test)[:,1])
    logit_roc_auc = auc(recall,precision)
    plt.figure()
    plt.plot(recall, precision, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 0],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve with SMOTED dataset')
    plt.legend(loc="lower right")
    #plt.savefig('Log_PR')
    plt.show()

def evaluate_model(X, y, repeats):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=repeats, random_state=seed)
    model = LogisticRegression()
    scores = cross_val_score(model, X, y, scoring='f1', cv=cv, n_jobs=-1)
    return scores

def f1_score_CV_estimates(X,y,repeats):
	results = list()
	for r in repeats:
		# evaluate using a given number of repeats
		scores = evaluate_model(X, y.values.ravel(), r)
	# summarize
		print('>%d mean=%.4f se=%.3f' % (r, np.mean(scores), np.std(scores)))
		# store
		results.append(scores)
	plt.boxplot(results, labels=[str(r) for r in repeats], showmeans=True)
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
