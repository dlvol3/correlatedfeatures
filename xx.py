# %% Test feature sets crafting
# Import the lib.
from scipy.linalg import cholesky
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, norm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
import os
from scipy import interp
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

# %% Creating the correlation references
# number of features in Set 1 & 2 no_total
no_total = 100
fs_sb = [10, 30, 60, 90]
for fi in range(len(fs_sb)):
    no_fs1 = fs_sb[fi]
    # in the for loop,
    no_fs2 = no_total - no_fs1
    m1 = np.full((no_fs1, no_fs1), 0.8)
    np.fill_diagonal(m1, 1)
    m2 = np.full((no_fs2, no_fs2), 0.1)
    np.fill_diagonal(m2, 1)
    m3 = np.full((30, 30), 0.05)
    np.fill_diagonal(m3, 1)
    # Calculating the3cholesky's decomposition matrix for the correlated set fs1
    uppercho1 = cholesky(m1, lower=False)
    uppercho2 = cholesky(m2, lower=False)
    uppercho3 = cholesky(m3, lower=False)
    # sample two var from different distributions
    # Not too far away, better to have a bit overlap
    var1 = np.random.normal(1, 1.0, size=(1000, no_fs1))
    var2 = np.random.normal(-1, 1.0, size=(1000, no_fs2))
    # Noise feature set
    var3 = np.random.normal(0, 0.5, size=(1000, 30))
    # Generating the correlated feature set fs1
    fs1 = var1 @ uppercho1
    fs2 = var2 @ uppercho2
    fs3 = var3 @ uppercho3
    plt.figure(figsize=(13, 5))
    for array in [fs1, fs2]:
        corr_m = list()
        a = [(i, j) for i in range(array.shape[1]) for j in range(i + 1, array.shape[1])]
        for i in range(len(a)):
            corr, _ = pearsonr(array[:, a[i][0]], array[:, a[i][1]])
            corr_m.append(corr)
        re = [abs(x) for x in corr_m]
        print(sum(re) / len(re))
        sns.distplot(corr_m, label="Distributions of pairwise correlations within each feature set")
    fn = "Distributions of pairwise correlations within each feature set when no_fs1 equals " + str(no_fs1) + ".png"
    plt.savefig(fn)
    plt.show()
    plt.figure(figsize=(13, 5))
    la = [fs1, fs2, fs3]
    lalist = ["fs1", "fs2", "fs3"]
    for i in range(len(la)):
        sns.distplot(la[i].flatten(), label=["Distributions of all values in " + lalist[i]])
    plt.legend()
    fn = "Distributions of all values in each of the feature set when no_fs1 equals " + str(no_fs1) + ".png"
    plt.savefig(fn)
    plt.show()
    # Fitting to norm. distributions for set 1&2
    mu1, std1 = norm.fit(fs1.flatten())
    mu2, std2 = norm.fit(fs2.flatten())
    # Get the symmetric point
    sym_p = round((mu1 + mu2) / 2, 3)
    # Merging arrays into dataframe
    fs1df = pd.DataFrame(data=fs1,
                         index=["Sample" + str(j) for j in range(1000)],
                         columns=["fs1_feature" + str(i) for i in range(no_fs1)])
    fs2df = pd.DataFrame(data=fs2,
                         index=["Sample" + str(j) for j in range(1000)],
                         columns=["fs2_feature" + str(i) for i in range(no_fs2)])
    fs3df = pd.DataFrame(data=fs3,
                         index=["Sample" + str(j) for j in range(1000)],
                         columns=["fs3_feature" + str(i) for i in range(30)])
    training = pd.concat([fs1df, fs2df, fs3df], axis=1)
    # Crafting the class of 1000 samples with bais(fs1 more importance than fs2)
    training['class'] = np.where(training.iloc[:, 0:no_fs1].sum(axis=1) / no_fs1 * 3 +
                                 training.iloc[:, no_fs1:no_total].sum(axis=1) / no_fs2 * 2 > sym_p, 1,
                                 0)
    # Check the class distribution
    sns.countplot(training['class'])
    fn = "fs1 more important_class distribution when no_fs1 equals " + str(no_fs1) + ".png"
    plt.savefig(fn)
    plt.show()
    # training RF
    # training.shape[1]
    x = training.iloc[:, 0:(training.shape[1] - 2)].values  # Features for training
    y = training.iloc[:, training.shape[1] - 1].values  # Labels of training
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123)
    rf1 = RandomForestClassifier(n_estimators=1000, criterion='gini', max_features="sqrt",
                                 oob_score=True, n_jobs=13, max_depth=15,
                                 verbose=0)
    rf1.fit(x_train, y_train)
    # Checking Performance
    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=6)
    classifier = rf1
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    for train, test in cv.split(x, y):
        probas_ = classifier.fit(x[train], y[train]).predict_proba(x[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic CV')
    plt.legend(loc="lower right")
    fn = "fs1 more important_ROC of 6-fold cross validation when no_fs1 equals " + str(no_fs1) + ".png"
    plt.savefig(fn)
    plt.show()
    # %%
    featurelist = training.iloc[:, 0:(training.shape[1] - 2)].columns.values.tolist()
    indexBR = list(range(len(featurelist)))
    featureimp = rf1.feature_importances_
    feature_importances = pd.DataFrame({'feature': featurelist, 'importance': featureimp})
    feature_importances = feature_importances.sort_values('importance', ascending=False)
    fn = "fs1 more important_feature importance when no_fs1 equals " + str(no_fs1) + ".txt"
    feature_importances.to_csv(fn, sep='\t')
    #####
    # Changing the classify equation, now fs2 contributes more than fs1
    #####
    training['class'] = np.where(training.iloc[:, 0:no_fs1].sum(axis=1) / no_fs1 * 2 +
                                 training.iloc[:, no_fs1:no_total].sum(axis=1) / no_fs2 * 3 > sym_p, 1,
                                 0)
    # Check the class distribution
    sns.countplot(training['class'])
    fn = "fs2 more important_class distribution when no_fs1 equals " + str(no_fs1) + ".png"
    plt.savefig(fn)
    plt.show()
    # training RF
    # training.shape[1]
    x = training.iloc[:, 0:(training.shape[1] - 2)].values  # Features for training
    y = training.iloc[:, training.shape[1] - 1].values  # Labels of training
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123)
    rf1 = RandomForestClassifier(n_estimators=1000, criterion='gini', max_features="sqrt",
                                 oob_score=True, n_jobs=13, max_depth=15,
                                 verbose=0)
    rf1.fit(x_train, y_train)
    # Checking Performance
    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=6)
    classifier = rf1
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    for train, test in cv.split(x, y):
        probas_ = classifier.fit(x[train], y[train]).predict_proba(x[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic CV')
    plt.legend(loc="lower right")
    fn = "fs2 more important_ROC of 6-fold cross validation when no_fs1 equals " + str(no_fs1) + ".png"
    plt.savefig(fn)
    plt.show()
    # %%
    featurelist = training.iloc[:, 0:(training.shape[1] - 2)].columns.values.tolist()
    indexBR = list(range(len(featurelist)))
    featureimp = rf1.feature_importances_
    feature_importances = pd.DataFrame({'feature': featurelist, 'importance': featureimp})
    feature_importances = feature_importances.sort_values('importance', ascending=False)
    fn = "fs2 more important_feature importance when no_fs1 equals " + str(no_fs1) + ".txt"
    feature_importances.to_csv(fn, sep='\t')
