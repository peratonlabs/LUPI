import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sys
import argparse
import lupi_svm as lupi

parser = argparse.ArgumentParser()

parser.add_argument("--nJobs", help="number of processors available", default=4, type=int)
parser.add_argument("--dataset", help="the dataset file", default='parkinsons.data', type=str)
parser.add_argument("--cv_score", help="cv_score: f1, accuracy", default='accuracy', type=str)
parser.add_argument("--cv_kfold", help="number of folds for cross validation", default=6, type=int)
parser.add_argument("--test_size", help="test_size: 0.25, 0.3, 0.33", default=0.33, type=float)
parser.add_argument("--pri_features", nargs='+', help="privileged features",
#                    default=[])
                    default=[16, 17, 18, 19, 20, 21])

args = parser.parse_args()
nJobs = args.nJobs

test_size = args.test_size

# The dataset file name is input by --dataset argument
# We use the UCI parkinsons dataset as an example
# The dataset can be download with the following command. run the command from current directory
# wget https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data
dataset_file = args.dataset

# load the dataset from the dataset file to pandas data frmae
print("... Loading the dataset: {}.".format(dataset_file))
data_set = pd.read_csv(dataset_file, header=0, delim_whitespace=False, sep=',')

# get the true label values y.  the "status" column is the label.
y = data_set.loc[:, 'status']

# To get the X, we need to clean the data a bit.
# drop the "status" label column.
# also drop the "name" column which is not used.
X = data_set.drop(['name', 'status'], axis=1)

# now X is the dataset with standard and privileged features.
# We take the non-linear features as privileged.
# [16, 17, 18, 19, 20, 21] are the privileged feature columns (0 index based)
# We use pri_features for the privileged feature list
pri_features = list(map(int, args.pri_features))

print("... Spliting the dataset and setup training and test dataset ...")
# Now split the dataset to training and test sets
X_train, X_test_all, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)


# set the privileged features in the test set to 0 so that the privileged information is removed
X_test = X_test_all.copy()
X_test.iloc[:, pri_features] = 0

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

# Create the lupi classifer, train with training set and predict on the test set.
lupi_clf = lupi.Composer(pri_features=pri_features,
                         krr_param_grid=krr_param_grid,
                         svc_param_grid=svc_param_grid,
                         nJobs=nJobs,
                         cv=cv_kfold,
                         cv_score=cv_score,
                         random_state=0)

print("... model training ...")
lupi_clf.fit(X_train, y_train)

print("... predicting on test data...")
y_pred = lupi_clf.predict(X_test, type='ln')

score = accuracy_score(y_test, y_pred)

print("...The accuracy performance on the test data is: \n   {}.".format(score))
