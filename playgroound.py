from scipy.linalg import cholesky
import numpy as np
import pandas as pd

#%% Corr.mat 1 & 2
m1 = np.full((50,50),0.85)
np.fill_diagonal(m1, 1)

m2 = np.full((50,50),0.2)
np.fill_diagonal(m2, 1)

uppercho1 = cholesky(m1, lower = False)
uppercho2 = cholesky(m2, lower=False)

# sample two var from different distributions

var1 = np.random.normal(1, 1.0, size=(1000, 50))
var2 = np.random.normal(-1, 1.0, size=(1000, 50))

# Calculate for the correlated features

fs1 = var1 @ uppercho1
fs2 = var2 @ uppercho2

#%% Check the correlation within and between feature sets

from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

for array in [fs1, fs2]:
    corr_m = list()
    a=[(i, j) for i in range(array.shape[1]) for j in range(i+1, array.shape[1])]
    for i in range(len(a)):
        corr, _ = pearsonr(array[:, a[i][0]], array[:, a[i][1]])
        corr_m.append(corr)
    re = [abs(x) for x in corr_m]
    print(sum(re)/len(re))
    plt.figure(figsize = (20,5))
    sns.distplot(corr_m)
    plt.show()

# Create the training set

fs1df = pd.DataFrame(data=fs1,
                     index=["Sample" + str(j) for j in range(1000)],
                     columns=["fs1_feature" + str(i) for i in range(50)])

fs2df = pd.DataFrame(data=fs2,
                     index=["Sample" + str(j) for j in range(1000)],
                     columns=["fs2_feature" + str(i) for i in range(50)])

training = pd.concat([fs1df, fs2df], axis = 1)

training['class'] = np.where(training.iloc[:, 0:50].sum(axis=1) * 3 +
                             training.iloc[:, 50:100].sum(axis=1) * 2 > 0, "1",
                             "0")
#%% train a RF

# Check the class distribution
sns.countplot(training['class'])
plt.show()

# 250 0 vs. 700 1
training.shape[1]
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
import os
x = training.iloc[:, 0:(training.shape[1] - 2)].values   # Features for training
y = training.iloc[:, training.shape[1]-1].values  # Labels of training

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=123)

rf1 = RandomForestClassifier(n_estimators=1000, criterion='gini', max_features="sqrt",
                             oob_score=True, n_jobs=13, max_depth=15,
                             verbose=0)

rf1.fit(x_train, y_train)

featurelist = training.iloc[:, 0:(training.shape[1] - 2)].columns.values.tolist()
indexBR = list(range(len(featurelist)))
featureimp = rf1.feature_importances_
feature_importances = pd.DataFrame({'feature': featurelist, 'importance': featureimp})
feature_importances.to_csv('featureimptest.txt', sep='\t')

os.getcwd()