from __future__ import division
import pandas as pd
import time, sys
from sklearn.model_selection import train_test_split
import argparse
import lupi_svm as lupi

parser = argparse.ArgumentParser()

parser.add_argument("--nJobs", help="number of processors available", default=4, type=int)
parser.add_argument("--nIter", help="number of iterations", default=20, type=int)
parser.add_argument("--dataset", help="the dataset file", default='parkinsons.data', type=str)
parser.add_argument("--cv_score", help="cv_score: f1, accuracy", default='accuracy', type=str)
parser.add_argument("--cv_kfold", help="number of folds for cross validation", default=6, type=int)
parser.add_argument("--test_size", help="test_size: 0.25, 0.3, 0.33", default=0.33, type=float)
parser.add_argument("--pri_features", nargs='+', help="privileged features",
                    default=[16, 17, 18, 19, 20, 21])

args = parser.parse_args()

nJobs = args.nJobs
nIter = args.nIter
dataset = args.dataset
cv_score = args.cv_score
test_size = args.test_size
pri_features = args.pri_features

data_set = pd.read_csv('parkinsons.data', header=0, delim_whitespace=False, sep=',')

# Setup the data sets.
# the "status" column is the label
y = data_set.loc[:, 'status']

# To get the X, we need to clean the data a bit.
# drop the "status" label column.
# also drop the "name" column which is not used.
X = data_set.drop(['name', 'status'], axis=1)

# reset the columns index
X.columns = range(X.shape[1])

# now X is the dataset with standard and privileged features.
# We take the non-linear features as privileged.
# [16, 17, 18, 19, 20, 21] are the privileged feature columns (0 index based)
# We use pri_features for the privileged feature list
pri_features = list(map(int, args.pri_features))

# Sanity check on the privileged feature list
if len(pri_features) <= 0:
    print("Warning: the number of privileged features is {}!!".format(len(pri_features)))
if len(pri_features) != len(set(pri_features)):
    print("The given pri_features has duplicates. It is {}!!".format(pri_features))
    sys.exit(1)
if min(pri_features) <= 0 or max(pri_features) >= X.shape[1]:
    print("The given pri_features is out of the range of X_train indices. It is {}!!".format(pri_features))
    sys.exit(1)
for i in pri_features:
    if not isinstance(i, int):
        print("The given pri_features has non integer element. It is {}!!".format(pri_features))
        sys.exit(1)

# set up the lupi regressor grid search parameters
regr_param_grid_dict = dict(gamma_list=[-2.0, 6.5, 0.4], C_list=[-2.5, 6.0, 0.4])
# set up the svm grid search parameters
svc_param_grid_dict = dict(gamma_list=[-2.0, 6.5, 0.4], C_list=[-2.5, 6.0, 0.4])
# set up cross validation KFlod and score function
cv_kfold = args.cv_kfold
cv_score = args.cv_score

app = lupi.App(svc_param_grid_dict=svc_param_grid_dict,
               regr_param_grid_dict=regr_param_grid_dict)

krr_param_grid = app.get_krr_param_grid()
svc_param_grid = app.get_svc_param_grid()

app.print_runconfig()

nseed = nIter
seeds = list(range(0, nseed, 1))

for ran_seed in seeds:
    loopstart = time.time()
    print("... running dataset split with random seed {} ... ".format(ran_seed))
    app.fp_ratio[ran_seed] = {}
    app.fn_ratio[ran_seed] = {}
    app.ac_ratio[ran_seed] = {}
    app.ac_err_ratio[ran_seed] = {}
    app.f1_score[ran_seed] = {}

    X_train, X_test_all, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=ran_seed)

    # set the privileged features in the test set to 0 so that the privileged information is removed
    X_test = X_test_all.copy()
    X_test.iloc[:, pri_features] = 0

    lupi_clf = lupi.Composer(pri_features=pri_features,
                             krr_param_grid=app.krr_param_grid,
                             svc_param_grid=app.svc_param_grid,
                             nJobs=nJobs,
                             cv=cv_kfold,
                             cv_score=cv_score,
                             random_state=ran_seed)
    lupi_clf.fit(X_train, y_train)
    cv_results = lupi_clf.predict(X_test, type='all')

    for key, value in cv_results.items():
        app.add_perf(ran_seed, y_test, value[0], type=key)

    # -- standard sklearn SVC on test dataset without privileged features--
    y_pred_std = lupi_clf.std_svm(X_train, y_train, X_test)
    app.add_perf(ran_seed, y_test, y_pred_std, type='STD')

    # --- on test dataset with full  privileged features ---
    y_pred_priv = lupi_clf.priv_svm(X_train, y_train, X_test_all)
    app.add_perf(ran_seed, y_test, y_pred_priv, type='PRIV')

    app.print_results()
    looptime = time.time() - loopstart
    print("Time Elapsed: %d seconds" % looptime)

app.print_results(final=True)
