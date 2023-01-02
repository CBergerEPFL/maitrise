import os
import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
import seaborn as sn
import pickle
import statsmodels.api as sm
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
    multilabel_confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_val_score,
    train_test_split,
    StratifiedKFold,
)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

sys.path.append(os.path.join(os.getcwd(), ".."))
from shared_utils.Custom_Logit import Logit_binary
from skfeature.function.information_theoretical_based import LCSI

seed = 0

list_name_features = [
    "Corr_interlead",
    "Corr_intralead",
    "wPMF",
    "SNRECG",
    "HR",
    "Kurtosis",
    "Flatline",
    "TSD",
]

dico_T_opt = {
    "Corr_interlead": 0.39,
    "Corr_intralead": 0.67,
    "wPMF": 0.116,
    "SNRECG": 0.48,
    "Kurtosis": 2.16,
    "Flatline": 0.51,
    "TSD": 0.42,
}

save_path = "/workspaces/maitrise/results"


def save_model_LR(X_data, y_data, cols, opp, **kwargs):
    if cols is None:
        print("Using : Backward_model_selection")
        cols = Backward_model_selection(X_data, y_data)
        if "HR" in cols and len(cols) > 1:
            Hindex = list(X_data[cols].columns.values).index("HR")
            model = Logit_binary(HR_index=Hindex, random_state=seed)
        else:
            Hindex = None
            model = LogisticRegression(random_state=seed)
        X = X_data[cols].values
        y = y_data.values
    else:
        if "HR" in cols and len(cols) > 1:
            Hindex = list(X_data[cols].columns.values).index("HR")
            model = Logit_binary(HR_index=Hindex, random_state=seed)
        else:
            Hindex = None
            model = LogisticRegression(random_state=seed)
        X = X_data[cols].values
        y = y_data.values

    # cv = StratifiedKFold(n_splits=10, random_state=True, shuffle=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y.ravel())

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sn.heatmap(cm, annot=True, annot_kws={"size": 16}, fmt="g")
    print(classification_report(y_test, y_pred))

    if kwargs.get("Model_name"):
        name_model = kwargs["Model_name"]
    else:
        if Hindex is not None and not opp:
            name_model = "Logit_bin_"
            for i in cols:
                name_model += i + "_"
        elif Hindex is not None and opp:
            name_model = "Logit_bin_"
            for i in cols:
                name_model += i + "_"
            name_model += "inverselabel"
        elif Hindex is None and not opp:
            name_model = "LogisticRegression"
            for i in cols:
                name_model += i + "_"
        else:
            name_model = "LogisticRegression"
            for i in cols:
                name_model += i + "_"
            name_model += "inverselabel"

    print(
        "This model will be saved at {} with the name : {}".format(
            save_path, name_model
        )
    )
    Model_folder = os.path.join(save_path, "Models")
    if not os.path.exists(Model_folder):
        os.mkdir(Model_folder)
    filename = name_model + ".sav"
    pickle.dump(model, open(os.path.join(Model_folder, filename), "wb"))


def Index_ML_calculator(original_label, prob_predicted):
    AUCROC = np.empty([10, 2])
    AUCPR = np.empty([10, 2])
    prec = np.empty([10, 2])
    rec = np.empty([10, 2])
    F1 = np.empty([10, 2])
    Spec = np.empty([10, 2])
    ACC = np.empty([10, 2])
    MCC = np.empty([10, 2])
    T = np.empty([10, 2])
    for i, y_ori in enumerate(original_label):

        rocauc_0, rocauc_1 = roc_auc_score(
            1 - y_ori, prob_predicted[i][:, 0]
        ), roc_auc_score(y_ori, prob_predicted[i][:, 1])
        AUCROC[i, 0], AUCROC[i, 1] = rocauc_0, rocauc_1
        precision_1, recall_1, threshold_1 = precision_recall_curve(
            y_ori, prob_predicted[i][:, 1], pos_label=1
        )
        precision_0, recall_0, threshold_0 = precision_recall_curve(
            y_ori, prob_predicted[i][:, 0], pos_label=0
        )
        AUCPR[i, 0], AUCPR[i, 1] = auc(recall_0, precision_0), auc(
            recall_1, precision_1
        )
        MCC_0, MCC_1 = np.array(
            [
                matthews_corrcoef(1 - y_ori, (prob_predicted[i][:, 0] >= t).astype(int))
                for t in threshold_0
            ]
        ), np.array(
            [
                matthews_corrcoef(y_ori, (prob_predicted[i][:, 1] >= t).astype(int))
                for t in threshold_1
            ]
        )
        MCC[i, 0], MCC[i, 1] = np.max(MCC_0), np.max(MCC_1)
        i_1 = threshold_1[np.argmax(MCC_1)]
        i_0 = threshold_0[np.argmax(MCC_0)]
        T[i, 0], T[i, 1] = i_0, i_1
        y_pred = (prob_predicted[i][:, 1] >= i_1).astype(int)
        precision, recall, f1, specificity, _, acc, _ = roc_pr_curve_multilabel(
            y_ori, y_pred
        )
        prec[i, 0], prec[i, 1] = precision[0], precision[1]
        rec[i, 0], rec[i, 1] = recall[0], recall[1]
        F1[i, 0], F1[i, 1] = f1[0], f1[1]
        Spec[i, 0], Spec[i, 1] = specificity[0], specificity[1]
        ACC[i, 0], ACC[i, 1] = acc[0], acc[1]

    df = pd.DataFrame(index=["0", "1"])

    df["Precision (mean,std) (Max MCC)"] = [
        (
            np.around(prec[:, 0].mean(), 2),
            np.around(prec[:, 0].std(), 2),
        ),
        (
            np.around(prec[:, 1].mean(), 2),
            np.around(prec[:, 1].std(), 2),
        ),
    ]
    df["Recall (mean,std) (Max MCC)"] = [
        (
            np.around(rec[:, 0].mean(), 2),
            np.around(rec[:, 0].std(), 2),
        ),
        (
            np.around(rec[:, 1].mean(), 2),
            np.around(rec[:, 1].std(), 2),
        ),
    ]
    df["Specificity (TNR) (mean,std) (Max MCC)"] = [
        (
            np.around(Spec[:, 0].mean(), 2),
            np.around(Spec[:, 0].std(), 2),
        ),
        (
            np.around(Spec[:, 1].mean(), 2),
            np.around(Spec[:, 1].std(), 2),
        ),
    ]
    df["F1 score (mean,std) (Max MCC)"] = [
        (
            np.around(F1[:, 0].mean(), 2),
            np.around(F1[:, 0].std(), 2),
        ),
        (
            np.around(F1[:, 1].mean(), 2),
            np.around(F1[:, 1].std(), 2),
        ),
    ]
    df["Accuracy (mean,std) (Max MCC)"] = [
        (
            np.around(ACC[:, 0].mean(), 2),
            np.around(ACC[:, 0].std(), 2),
        ),
        (
            np.around(ACC[:, 1].mean(), 2),
            np.around(ACC[:, 1].std(), 2),
        ),
    ]
    df["MCC (mean,std) (Max value)"] = [
        (
            np.around(MCC[:, 0].mean(), 2),
            np.around(MCC[:, 0].std(), 2),
        ),
        (
            np.around(MCC[:, 1].mean(), 2),
            np.around(MCC[:, 1].std(), 2),
        ),
    ]
    df["AUC ROC (mean,std)"] = [
        (
            np.around(AUCROC[:, 0].mean(), 2),
            np.around(AUCROC[:, 0].std(), 2),
        ),
        (
            np.around(AUCROC[:, 1].mean(), 2),
            np.around(AUCROC[:, 1].std(), 2),
        ),
    ]
    df["AUC PR (mean,std)"] = [
        (
            np.around(AUCPR[:, 0].mean(), 2),
            np.around(AUCPR[:, 0].std(), 2),
        ),
        (
            np.around(AUCPR[:, 1].mean(), 2),
            np.around(AUCPR[:, 1].std(), 2),
        ),
    ]
    df["Optimal Threshold using Maximum  MCC"] = [
        (np.around(T[:, 0].mean(), 2), np.around(T[:, 0].std(), 2)),
        (np.around(T[:, 1].mean(), 2), np.around(T[:, 1].std(), 2)),
    ]
    return df


def Classification_report_model(X_data, y_data, cols, **kwargs):
    if cols is None:
        print("Using : Backward_model_selection")
        cols = Backward_model_selection(X_data, y_data)
        if "HR" in cols and len(cols) > 1:
            Hindex = list(X_data[cols].columns.values).index("HR")
            model = Logit_binary(HR_index=Hindex, random_state=seed)
        else:
            Hindex = None
            model = LogisticRegression(random_state=seed)
        X = X_data[cols].values
        y = y_data.values
    else:
        if "HR" in cols and len(cols) > 1:
            Hindex = list(X_data[cols].columns.values).index("HR")
            model = Logit_binary(HR_index=Hindex, random_state=seed)
        else:
            Hindex = None
            model = LogisticRegression(random_state=seed)
        X = X_data[cols].values
        y = y_data.values

    cv = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
    original_label = []
    prob_predicted_model = []

    for i, (train, test) in enumerate(cv.split(X, y.ravel())):
        model.fit(X[train], y.ravel()[train])
        original_label.append(y.ravel()[test])
        prob_predicted_model.append(model.predict_proba(X[test]))
    df = Index_ML_calculator(original_label, prob_predicted_model)
    display(df)

    ##For the index alone without model
    columns_remove = np.array([])
    for j in range(X.shape[1]):
        if not (np.min(X[:, j]) >= 0 and np.max(X[:, j]) <= 1):
            columns_remove = np.append(columns_remove, j)
            print(
                "The features ",
                np.array(cols)[columns_remove.astype(int)],
                "will be removed as they are not between 0 and 1",
            )
            del cols[j]
    if len(columns_remove) > 0:
        X = np.delete(X, columns_remove.astype(int), axis=1)
    else:
        print("No features were removed!")
    if len(cols) == 0:
        print("No features remaining.")
    else:
        for i, c in enumerate(cols):
            original_lab = []
            prob_pred_ind = []
            X_s = X[:, i]
            for i, (train, test) in enumerate(cv.split(X_s, y.ravel())):
                original_lab.append(y.ravel()[test])
                pos_pred = X_s[test]
                neg_pred = 1 - pos_pred
                pred_prob = np.c_[neg_pred, pos_pred]
                prob_pred_ind.append(pred_prob)

            df_p = Index_ML_calculator(original_lab, prob_pred_ind)
            print(f"{c} alone:")
            display(df_p)


def ROC_PR_CV_curve_model(
    X_data,
    y_data,
    cols=None,
    pos_label=1,
    k_cv=10,
    model_type="Logistic",
    Feature_selection="Backward Model Selection",
):
    if cols is None:
        print("Using : {}".format(Feature_selection))
        cols = Backward_model_selection(X_data, y_data)
        if "HR" in cols and len(cols) > 1:
            Hindex = list(X_data[cols].columns.values).index("HR")

        else:
            Hindex = None
        X = X_data[cols].values
        y = y_data.values
    else:
        if "HR" in cols and len(cols) > 1:
            Hindex = list(X_data[cols].columns.values).index("HR")
        else:
            Hindex = None

        X = X_data[cols].values
        y = y_data.values

    if model_type == "ExtraTreeClassifier":
        model = ExtraTreesClassifier(random_state=seed)
    elif model_type == "RandomTreeClassifier":
        model = RandomForestClassifier(random_state=seed)
    elif model_type == "Logistic" and Hindex is not None:
        model_type = model_type + " Binary"
        model = Logit_binary(HR_index=Hindex, random_state=seed)
    else:
        model = LogisticRegression(random_state=seed)

    print("Model Performance : ")
    plt.figure()
    plot_graph_ROC_PR_model(X, y, cols, model, pos_label, k_cv, model_type)

    print("Performance of each indexes alone:")
    plt.figure()
    plot_graph_ROC_PR_indexes(X, y, cols, pos_label, k_cv)


def plot_graph_ROC_PR_model(
    X, y, cols, model, pos_label=1, k_cv=10, model_type="Logistic"
):
    pos_lab = pos_label
    cv = StratifiedKFold(n_splits=k_cv)
    mean_fpr = np.linspace(0, 1, 500)
    mean_recall = np.linspace(0, 1, 500)
    tprs = []
    precs = []
    aucs_roc = []
    aucs_pr = []
    arr_coeff = np.empty([k_cv, len(cols)])
    for i, (train, test) in enumerate(cv.split(X, y.ravel())):
        model.fit(X[train], y[train].ravel())
        y_score = model.predict_proba(X[test])
        arr_coeff[i, :] = model.coef_[0]

        fpr, tpr, _ = roc_curve(y[test], y_score[:, pos_lab], pos_label=pos_lab)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0
        tprs.append(interp_tpr)
        aucs_roc.append(auc(fpr, tpr))

        precision, recall, _ = precision_recall_curve(
            y[test], y_score[:, pos_lab], pos_label=pos_lab
        )
        index_rec = np.argsort(recall)
        interp_prec = np.interp(mean_recall, np.sort(recall), precision[index_rec])
        # interp_prec[0] = 1
        precs.append(interp_prec)
        aucs_pr.append(auc(recall, precision))

    mean_coeff = arr_coeff.mean(axis=0)
    sd_coeff = arr_coeff.std(axis=0)
    for count, values in enumerate(cols):
        print(values, "coefficients : ", mean_coeff[count], "+-", sd_coeff[count])

    precision_avg = np.mean(precs, axis=0)
    # mean_recall = np.mean(recs,axis = 0)

    mean_auc_pr = np.mean(aucs_pr)  # auc(mean_fpr, mean_tpr)
    std_auc_pr = np.std(aucs_pr)
    std_precs = np.std(precs, axis=0)
    precs_upper = np.minimum(precision_avg + std_precs, 1)
    precs_lower = np.maximum(precision_avg - std_precs, 0)

    tpr_avg = np.mean(tprs, axis=0)
    # mean_fpr = np.mean(fprs,axis = 0)
    mean_auc_roc = np.mean(aucs_roc)  # auc(mean_fpr, mean_tpr)
    std_auc_roc = np.std(aucs_roc)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(tpr_avg + std_tpr, 1)
    tprs_lower = np.maximum(tpr_avg - std_tpr, 0)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 25))
    color = iter(plt.cm.rainbow(np.linspace(0, 1, k_cv)))

    ax[0].plot(
        [0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8
    )
    ax[1].plot(
        [0, 1], [0, 0], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8
    )

    for j in range(k_cv):
        c = next(color)
        ax[0].plot(
            mean_fpr,
            tprs[j],
            label="ROC fold {} with AUC = {:.2f}".format(j, aucs_roc[j]),
            color=c,
            alpha=0.3,
            lw=1,
        )
        ax[1].plot(
            mean_recall,
            precs[j],
            label="PR fold {} with AUC = {:.2f}".format(j, aucs_pr[j]),
            color=c,
            alpha=0.3,
            lw=1,
        )

    ax[0].plot(
        mean_fpr,
        tpr_avg,
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc_roc, std_auc_roc),
        color="b",
    )
    ax[0].fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )
    ax[0].set_xlabel("FPR")
    ax[0].set_ylabel("TPR")
    ax[0].set_title(f"ROC curve for {model_type} using {cols}")
    ax[0].grid()
    ax[0].legend(loc="best")
    ax[1].plot(
        mean_recall,
        precision_avg,
        label=r"Mean PR (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc_pr, std_auc_pr),
        color="b",
    )
    ax[1].fill_between(
        mean_recall,
        precs_lower,
        precs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].set_title(f"PR curve for {model_type} using {cols}")
    ax[1].grid()
    ax[1].legend(loc="best")

    plt.show()


def plot_graph_ROC_PR_indexes(X, y, cols, pos_label=1, k_cv=10):
    pos_lab = pos_label
    cv = StratifiedKFold(n_splits=k_cv)

    for j, col in enumerate(cols):
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 15))
        color = iter(plt.cm.rainbow(np.linspace(0, 1, k_cv)))
        mean_fpr = np.linspace(0, 1, 500)
        mean_recall = np.linspace(0, 1, 500)
        tprs = []
        precs = []
        aucs_roc = []
        aucs_pr = []
        X_s = X[:, j]
        if not (np.min(X_s) >= 0 and np.max(X_s) <= 1):
            raise AttributeError(
                "You cannot inverse the labeling if your index is not normalized!"
            )
        for i, (train, test) in enumerate(cv.split(X_s, y.ravel())):
            pos_score = X_s[test]
            neg_score = 1 - pos_score
            y_score = np.c_[neg_score, pos_score]

            fpr, tpr, _ = roc_curve(y[test], y_score[:, pos_lab], pos_label=pos_lab)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0
            tprs.append(interp_tpr)
            aucs_roc.append(auc(fpr, tpr))

            precision, recall, _ = precision_recall_curve(
                y[test], y_score[:, pos_lab], pos_label=pos_lab
            )
            index_rec = np.argsort(recall)
            interp_prec = np.interp(mean_recall, np.sort(recall), precision[index_rec])
            # interp_prec[0] = 1
            precs.append(interp_prec)
            aucs_pr.append(auc(recall, precision))

        precision_avg = np.mean(precs, axis=0)
        # mean_recall = np.mean(recs,axis = 0)

        mean_auc_pr = np.mean(aucs_pr)  # auc(mean_fpr, mean_tpr)
        std_auc_pr = np.std(aucs_pr)
        std_precs = np.std(precs, axis=0)
        precs_upper = np.minimum(precision_avg + std_precs, 1)
        precs_lower = np.maximum(precision_avg - std_precs, 0)

        tpr_avg = np.mean(tprs, axis=0)
        # mean_fpr = np.mean(fprs,axis = 0)
        mean_auc_roc = np.mean(aucs_roc)  # auc(mean_fpr, mean_tpr)
        std_auc_roc = np.std(aucs_roc)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(tpr_avg + std_tpr, 1)
        tprs_lower = np.maximum(tpr_avg - std_tpr, 0)

        ax[0].plot(
            [0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8
        )
        ax[1].plot(
            [0, 1], [0, 0], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8
        )

        for e in range(k_cv):
            c = next(color)
            ax[0].plot(
                mean_fpr,
                tprs[e],
                label="ROC fold {} with AUC = {:.2f}".format(e, aucs_roc[e]),
                color=c,
                alpha=0.3,
                lw=1,
            )
            ax[1].plot(
                mean_recall,
                precs[e],
                label="PR fold {} with AUC = {:.2f}".format(e, aucs_pr[e]),
                color=c,
                alpha=0.3,
                lw=1,
            )

        ax[0].plot(
            mean_fpr,
            tpr_avg,
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc_roc, std_auc_roc),
            color="b",
        )
        ax[0].fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )
        ax[0].set_xlabel("FPR")
        ax[0].set_ylabel("TPR")
        ax[0].set_title(f"ROC curve for {col}")
        ax[0].grid()
        ax[0].legend(loc="best")
        ax[1].plot(
            mean_recall,
            precision_avg,
            label=r"Mean PR (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc_pr, std_auc_pr),
            color="b",
        )
        ax[1].fill_between(
            mean_recall,
            precs_lower,
            precs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )
        ax[1].set_xlabel("Recall")
        ax[1].set_ylabel("Precision")
        ax[1].set_title(f"PR curve for {col}")
        ax[1].grid()
        ax[1].legend(loc="best")

        plt.show()


def Global_comp_ROC_PR_mean_curve(
    X_data,
    y_data,
    cols_models,
    models_name,
    k_cv=10,
    pos_label=1,
    model_type="Logistic",
):

    cv = StratifiedKFold(n_splits=k_cv)
    model_dict = {}
    y = y_data.values
    for count, col in enumerate(cols_models):
        if col is None:
            print("Using : Nackward model selection")
            col = Backward_model_selection(X_data, y_data)
            if "HR" in col and len(col) > 1:
                Hindex = list(X_data[col].columns.values).index("HR")
            else:
                Hindex = None
            X_model = X_data[col].values
        else:
            if "HR" in col and len(col) > 1:
                Hindex = list(X_data[col].columns.values).index("HR")
            else:
                Hindex = None

            X_model = X_data[col].values.copy()
        if model_type == "ExtraTreeClassifier":
            model = ExtraTreesClassifier(random_state=seed)
        elif model_type == "RandomTreeClassifier":
            model = RandomForestClassifier(random_state=seed)
        elif model_type == "Logistic" and Hindex is not None:
            model_type = model_type + " Binary"
            model = Logit_binary(HR_index=Hindex, random_state=seed)
        else:
            model = LogisticRegression(random_state=seed)
        model_dict[models_name[count]] = (model, X_model)

    cols_index = list(X_data.columns.values)
    X = X_data.values
    columns_remove = np.array([])
    for j in range(X.shape[1]):
        if not (np.min(X[:, j]) >= 0 and np.max(X[:, j]) <= 1):
            columns_remove = np.append(columns_remove, j)
            print(
                "The features ",
                np.array(cols_index)[columns_remove.astype(int)],
                "will be removed as they are not between 0 and 1",
            )
            del cols_index[j]
    if len(columns_remove) > 0:
        X = np.delete(X, columns_remove.astype(int), axis=1)

    else:
        print("No features were removed!")

    if len(cols_index) == 0:
        raise AttributeError("No features remaining!")

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 20))
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(cols_index) + len(models_name))))
    for j, col in enumerate(cols_index):
        c = next(color)
        mean_fpr = np.linspace(0, 1, 500)
        mean_recall = np.linspace(0, 1, 500)
        tprs = []
        precs = []
        aucs_roc = []
        aucs_pr = []
        X_s = X[:, j]
        if not (np.min(X_s) >= 0 and np.max(X_s) <= 1):
            raise AttributeError(
                "You cannot inverse the labeling if your index is not normalized!"
            )
        for i, (train, test) in enumerate(cv.split(X_s, y.ravel())):
            pos_score = X_s[test]
            neg_score = 1 - pos_score
            y_score = np.c_[neg_score, pos_score]

            fpr, tpr, _ = roc_curve(
                y[test].ravel(), y_score[:, pos_label], pos_label=pos_label
            )
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0
            tprs.append(interp_tpr)
            aucs_roc.append(auc(fpr, tpr))

            precision, recall, _ = precision_recall_curve(
                y[test], y_score[:, pos_label], pos_label=pos_label
            )
            index_rec = np.argsort(recall)
            interp_prec = np.interp(mean_recall, np.sort(recall), precision[index_rec])
            precs.append(interp_prec)
            aucs_pr.append(auc(recall, precision))

        precision_avg = np.mean(precs, axis=0)
        mean_auc_pr = np.mean(aucs_pr)
        std_auc_pr = np.std(aucs_pr)
        tpr_avg = np.mean(tprs, axis=0)
        mean_auc_roc = np.mean(aucs_roc)
        std_auc_roc = np.std(aucs_roc)

        ax[0].plot(
            mean_fpr,
            tpr_avg,
            color=c,
            label="Mean ROC curve {} : AUC = {:.2f} +- {:.2f}".format(
                col, mean_auc_roc, std_auc_roc
            ),
        )
        ax[1].plot(
            mean_recall,
            precision_avg,
            color=c,
            label="Mean PR curve {}: AUC = {:.2f} +- {:.2f}".format(
                col, mean_auc_pr, std_auc_pr
            ),
        )

    for m in models_name:
        c = next(color)
        model_m, X_m = model_dict[m][0], model_dict[m][1]
        mean_fpr = np.linspace(0, 1, 500)
        mean_recall = np.linspace(0, 1, 500)
        tprs = []
        precs = []
        aucs_roc = []
        aucs_pr = []
        for _, (train, test) in enumerate(cv.split(X_m, y.ravel())):
            model_m.fit(X_m[train], y[train].ravel())
            y_score = model_m.predict_proba(X_m[test])
            fpr, tpr, _ = roc_curve(
                y[test].ravel(), y_score[:, pos_label], pos_label=pos_label
            )
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0
            tprs.append(interp_tpr)
            aucs_roc.append(auc(fpr, tpr))

            precision, recall, _ = precision_recall_curve(
                y[test].ravel(), y_score[:, pos_label], pos_label=pos_label
            )
            index_rec = np.argsort(recall)
            interp_prec = np.interp(mean_recall, np.sort(recall), precision[index_rec])
            precs.append(interp_prec)
            aucs_pr.append(auc(recall, precision))
        precision_avg = np.mean(precs, axis=0)
        mean_auc_pr = np.mean(aucs_pr)
        std_auc_pr = np.std(aucs_pr)
        tpr_avg = np.mean(tprs, axis=0)
        mean_auc_roc = np.mean(aucs_roc)
        std_auc_roc = np.std(aucs_roc)
        ax[0].plot(
            mean_fpr,
            tpr_avg,
            color=c,
            label="Mean ROC curve {} : AUC = {:.2f} +- {:.2f}".format(
                m, mean_auc_roc, std_auc_roc
            ),
        )
        ax[1].plot(
            mean_recall,
            precision_avg,
            color=c,
            label="Mean PR curve {}: AUC = {:.2f} +- {:.2f}".format(
                m, mean_auc_pr, std_auc_pr
            ),
        )

    ax[0].plot([0, 1], [0, 1], "--k", label="Reference line")
    ax[0].set_xlabel("False Positive Rate")
    ax[0].set_ylabel("True Positive Rate")
    ax[0].set_title(f"Testing mean ROC Curve for all indexes created ")
    ax[0].legend(loc=4)
    ax[0].grid()

    ax[1].plot([0, 1], [0, 0], "--k", label="Reference line")
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].set_title(f"Testing mean PR Curve for all indexes created ")
    ax[1].legend(loc=4)
    ax[1].grid()
    plt.show()


def Backward_model_selection(X, y, threshold_out=0.001):
    initial_feature_set = list(X.columns.values)
    logit_model = sm.Logit(y.values.ravel(), X)
    result = logit_model.fit()
    sumsum = results_summary_to_dataframe(result)
    list_pval = np.array(sumsum["pvals"].values)
    max_pval = sumsum["pvals"].max()
    while max_pval >= threshold_out:
        idx_maxPval = np.array(initial_feature_set)[list_pval == max_pval]
        initial_feature_set.remove(idx_maxPval)
        logit_mod = sm.Logit(y, X[initial_feature_set])
        result = logit_mod.fit()
        sumsum = results_summary_to_dataframe(result)
        max_pval = sumsum["pvals"].max()
        list_pval = np.array(sumsum["pvals"].values)
    return initial_feature_set


def evaluate_model(X_data, y_data, repeats):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=repeats, random_state=seed)
    if "HR" in X_data.columns.values:
        index = list(X_data.columns.values).index("HR")
        X = X_data.values
        y = y_data.values.ravel()
        model = Logit_binary(index, random_state=seed)

    else:
        X = X_data.values
        y = y_data.values.ravel()
        model = LogisticRegression(random_state=seed)

    scores = cross_val_score(model, X, y, scoring="f1", cv=cv, n_jobs=-1)

    return scores


def f1_score_CV_estimates(X, y, repeats):
    results = list()
    for r in range(1, repeats):
        # evaluate using a given number of repeats
        scores = evaluate_model(X, y, r)
        # summarize
        print(">%d mean=%.4f se=%.3f" % (r, np.mean(scores), np.std(scores)))
        # store
        results.append(scores)
    plt.boxplot(results, labels=[str(r) for r in range(repeats)], showmeans=True)
    plt.show()


def results_summary_to_dataframe(results):
    """take the result of an statsmodel results table and transforms it into a dataframe"""
    pvals = results.pvalues
    coeff = results.params
    conf_lower = results.conf_int()[0]
    conf_higher = results.conf_int()[1]

    results_df = pd.DataFrame(
        {
            "pvals": pvals,
            "coeff": coeff,
            "conf_lower": conf_lower,
            "conf_higher": conf_higher,
        }
    )

    # Reordering...
    results_df = results_df[["coeff", "pvals", "conf_lower", "conf_higher"]]
    return results_df


def roc_pr_curve_multilabel(y_true, y_pred):
    mcm = multilabel_confusion_matrix(y_true, y_pred, labels=[0, 1])
    mcm_0 = mcm[0, :, :]
    mcm_1 = mcm[1, :, :]

    tn_0, fn_0, tp_0, fp_0 = mcm_0[0, 0], mcm_0[1, 0], mcm_0[1, 1], mcm_0[0, 1]
    tn_1, fn_1, tp_1, fp_1 = mcm_1[0, 0], mcm_1[1, 0], mcm_1[1, 1], mcm_1[0, 1]

    prec_0, prec_1 = tp_0 / (tp_0 + fp_0), tp_1 / (tp_1 + fp_1)
    rec_0, rec_1 = tp_0 / (tp_0 + fn_0), tp_1 / (tp_1 + fn_1)
    f1_0, f1_1 = 2 * (prec_0 * rec_0) / (prec_0 + rec_0), 2 * (prec_1 * rec_1) / (
        prec_1 + rec_1
    )
    fpr_0, fpr_1 = fp_0 / (fp_0 + tn_0), fp_1 / (fp_1 + tn_1)
    spec_0, spec_1 = 1 - fpr_0, 1 - fpr_1
    acc_0, acc_1 = (tp_0 + tn_0) / (tp_0 + tn_0 + fp_0 + fn_0), (tp_1 + tn_1) / (
        tp_1 + tn_1 + fp_1 + fn_1
    )

    MCC_0, MCC_1 = (tp_0 * tn_0 - fp_0 * fn_0) / (
        np.sqrt((tp_0 + fp_0) * (tp_0 + fn_0) * (tn_0 + fp_0) * (tn_0 + fn_0))
    ), (tp_1 * tn_1 - fp_1 * fn_1) / (
        np.sqrt((tp_1 + fp_1) * (tp_1 + fn_1) * (tn_1 + fp_1) * (tn_1 + fn_1))
    )

    return (
        np.array([prec_0, prec_1]),
        np.array([rec_0, rec_1]),
        np.array([f1_0, f1_1]),
        np.array([spec_0, spec_1]),
        np.array([fpr_0, fpr_1]),
        np.array([acc_0, acc_1]),
        np.array([MCC_0, MCC_1]),
    )


def ExtraTreeClassifier_CV_Feature_selection(X_data, y_data, k_cv=10):
    model = ExtraTreesClassifier(random_state=seed)
    cv = StratifiedKFold(n_splits=k_cv)
    cols = X_data.columns.values
    df = pd.DataFrame(index=X_data.columns)
    X = X_data.values
    y = y_data.values
    for i, (train, test) in enumerate(cv.split(X, y.ravel())):
        model.fit(X[train], y[train].ravel())
        feat_importances = pd.Series(model.feature_importances_, index=X_data.columns)
        df[f"{i+1} fold"] = feat_importances
    df_n = df.to_numpy()
    mean_val = np.mean(df_n, axis=1)
    std_val = np.std(df_n, axis=1)
    plt.figure()
    plt.bar(cols, mean_val)
    plt.errorbar(
        cols,
        mean_val,
        yerr=std_val,
        alpha=0.5,
        fmt="o",
        color="r",
        ecolor="black",
        capsize=10,
    )
    plt.title(
        f"Feature importance from ExtraTreeClassifier for {k_cv} Fold CV on training set"
    )
    plt.xlabel("Features")
    plt.ylabel("Gini Score")
    plt.grid()
    plt.tight_layout()
    plt.show()


def Kbest_MutulaInformation_CV(X_data, y_data, k_cv=10):
    model = SelectKBest(score_func=mutual_info_classif, k=len(X_data.columns.values))
    cv = StratifiedKFold(n_splits=k_cv)
    cols = X_data.columns.values
    df = pd.DataFrame(index=X_data.columns)
    X = X_data.values
    y = y_data.values
    for i, (train, test) in enumerate(cv.split(X, y.ravel())):
        fit = model.fit(X[train], y[train].ravel())
        df[f"{i+1} fold"] = pd.DataFrame(fit.scores_, index=cols)

    df_n = df.to_numpy()
    mean_val = np.mean(df_n, axis=1)
    std_val = np.std(df_n, axis=1)
    plt.figure()
    plt.bar(cols, mean_val)
    plt.errorbar(
        cols,
        mean_val,
        yerr=std_val,
        alpha=0.5,
        fmt="o",
        color="r",
        ecolor="black",
        capsize=10,
    )
    plt.title(f"Mutual information for {k_cv} Fold CV on training set")
    plt.xlabel("Features")
    plt.ylabel("Mutual Information")
    plt.grid()
    plt.tight_layout()
    plt.show()


###Link for the following code : https://github.com/jundongl/scikit-feature/blob/master/skfeature/function/information_theoretical_based/JMI.py


def jmi(X, y, **kwargs):
    """
    This function implements the JMI feature selection
    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data, guaranteed to be discrete
    y: {numpy array}, shape (n_samples,)
        input class labels
    kwargs: {dictionary}
        n_selected_features: {int}
            number of features to select
    Output
    ------
    F: {numpy array}, shape (n_features,)
        index of selected features, F[0] is the most important feature
    J_CMI: {numpy array}, shape: (n_features,)
        corresponding objective function value of selected features
    MIfy: {numpy array}, shape: (n_features,)
        corresponding mutual information between selected features and response
    Reference
    ---------
    Brown, Gavin et al. "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection." JMLR 2012.
    """
    if "n_selected_features" in kwargs.keys():
        n_selected_features = kwargs["n_selected_features"]
        F, J_CMI, MIfy = LCSI.lcsi(
            X, y, function_name="JMI", n_selected_features=n_selected_features
        )
    else:
        F, J_CMI, MIfy = LCSI.lcsi(X, y, function_name="JMI")
    return F, J_CMI, MIfy


def discretize_data(X_data):
    X_dis = np.zeros_like(X_data.values)
    for j in X_data.columns.values:
        i = list(X_data.columns.values).index(j)
        if j == "HR":
            X_dis[:, i] = X_data[j]
        else:
            X_dis[:, i] = np.digitize(X_data[j], bins=[dico_T_opt[j]])
    return X_dis


def JMI_calculator(X_data, y_data, k_cv=10):

    cv = StratifiedKFold(n_splits=k_cv, shuffle=True, random_state=seed)
    df_jmi_train = pd.DataFrame(index=X_data.columns)
    df_Fy_jmi_train = pd.DataFrame(index=X_data.columns)
    df_jmi_test = pd.DataFrame(index=X_data.columns)
    df_Fy_jmi_test = pd.DataFrame(index=X_data.columns)

    X_dis = discretize_data(X_data)
    for i, (train, test) in enumerate(cv.split(X_dis, y_data.values.ravel())):

        F_importance_train, F_JMI_train, Fy_JMI_train = jmi(
            X_dis[train],
            y_data.values[train].ravel(),
            n_selected_features=(len(X_data.columns.values)),
        )
        F_importance_test, F_JMI_test, Fy_JMI_test = jmi(
            X_dis[test],
            y_data.values[test].ravel(),
            n_selected_features=(len(X_data.columns.values)),
        )
        df_jmi_train[f"{i+1} fold"] = pd.DataFrame(
            F_JMI_train, index=X_data.columns.values[F_importance_train]
        )
        df_Fy_jmi_train[f"{i+1} fold"] = pd.DataFrame(
            Fy_JMI_train, index=X_data.columns.values[F_importance_train]
        )
        # print(F_importance_test)
        # print(pd.DataFrame(F_JMI_train,index=X_data.columns.values[F_importance_train]))
        df_jmi_test[f"{i+1} fold"] = pd.DataFrame(
            F_JMI_test, index=X_data.columns.values[F_importance_test]
        )
        df_Fy_jmi_test[f"{i+1} fold"] = pd.DataFrame(
            Fy_JMI_test, index=X_data.columns.values[F_importance_test]
        )

    df_jmi_train_n = df_jmi_train.to_numpy()
    df_Fy_jmi_train_n = df_Fy_jmi_train.to_numpy()
    df_jmi_test_n = df_jmi_test.to_numpy()
    df_Fy_jmi_test_n = df_Fy_jmi_test.to_numpy()
    mean_jmi_train = np.mean(df_jmi_train_n, axis=1)
    std_jmi_train = np.std(df_jmi_train_n, axis=1)
    mean_Fy_jmi_train = np.mean(df_Fy_jmi_train_n, axis=1)
    std_Fy_jmi_train = np.std(df_Fy_jmi_train_n, axis=1)
    mean_jmi_test = np.mean(df_jmi_test_n, axis=1)
    std_jmi_test = np.std(df_jmi_test_n, axis=1)
    mean_Fy_jmi_test = np.mean(df_Fy_jmi_test_n, axis=1)
    std_Fy_jmi_test = np.std(df_Fy_jmi_test_n, axis=1)
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    fig.tight_layout(h_pad=4)
    ax[0, 0].bar(X_data.columns.values, mean_jmi_train)
    ax[0, 0].errorbar(
        X_data.columns.values,
        mean_jmi_train,
        yerr=std_jmi_train,
        lw=2,
        capsize=10,
        capthick=2,
        color="r",
        ecolor="black",
        linestyle="",
    )
    ax[0, 0].set_ylabel("JMI value")
    ax[0, 0].set_title(
        "JMI between each features for a {} fold stratified CV from training set".format(
            k_cv
        )
    )
    ax[0, 0].grid()
    plt.setp(ax[0, 0].get_xticklabels(), rotation=30, horizontalalignment="right")
    ax[1, 0].bar(X_data.columns.values, mean_Fy_jmi_train)
    ax[1, 0].errorbar(
        X_data.columns.values,
        mean_Fy_jmi_train,
        yerr=std_Fy_jmi_train,
        lw=2,
        capsize=10,
        capthick=2,
        color="r",
        ecolor="black",
        linestyle="",
    )
    ax[1, 0].set_xlabel("Features")
    ax[1, 0].set_ylabel("MI value")
    ax[1, 0].set_title(
        "MI between selected features and response for a {} fold stratified CV from training set".format(
            k_cv
        )
    )
    ax[1, 0].grid()
    plt.setp(ax[1, 0].get_xticklabels(), rotation=30, horizontalalignment="right")
    ax[0, 1].bar(X_data.columns.values, mean_jmi_test)
    ax[0, 1].errorbar(
        X_data.columns.values,
        mean_jmi_test,
        yerr=std_jmi_test,
        lw=2,
        capsize=10,
        capthick=2,
        color="r",
        ecolor="black",
        linestyle="",
    )
    ax[0, 1].set_ylabel("JMI value")
    ax[0, 1].set_title(
        "JMI between each features for a {} fold stratified CV from testing set".format(
            k_cv
        )
    )
    ax[0, 1].grid()
    plt.setp(ax[0, 1].get_xticklabels(), rotation=30, horizontalalignment="right")
    ax[1, 1].bar(X_data.columns.values, mean_Fy_jmi_test)
    ax[1, 1].errorbar(
        X_data.columns.values,
        mean_Fy_jmi_test,
        yerr=std_Fy_jmi_test,
        lw=2,
        capsize=10,
        capthick=2,
        color="r",
        ecolor="black",
        linestyle="",
    )
    ax[1, 1].set_xlabel("Features")
    ax[1, 1].set_ylabel("MI value")
    ax[1, 1].set_title(
        "MI between selected features and response for a {} fold stratified CV from testing set".format(
            k_cv
        )
    )
    ax[1, 1].grid()
    plt.setp(ax[1, 1].get_xticklabels(), rotation=30, horizontalalignment="right")
    fig.subplots_adjust(top=1.10)


def Uniform_threshold(t, y):
    if (len(y) - len(t)) == 1:
        y = y[:1]
    mean_t = np.linspace(np.min(t), np.max(t), 500)
    t_new = np.interp(mean_t, y, t)
    return t_new
