## 1. LUPI SVM

This software is an implementation of LUPI (Learning Using Privileged Information) paradigm based on knowledge transfer approach.
This paradigm is applicable to the situation where some of the features of the training data (referred to as "privileged" features) are
absent in the test data, whereas other features (referred to as "standard" features) are present both in training and test data. Such a
situation can arise when privileged features represent the relevant information that is, neverthelss, difficult / costly / impossible to
collect for (online) test data, while one could afford to acquire such information for (offline) training data.

The knowledge transfer approach to LUPI constructs regressions of "privileged" features via "standard" ones. The current version of the software computes linear regression, kernel ridge regression and a decision function.  These "regressed" features (and the decision function feature) are then used
for making both training and test sets use the same set of features, thus enabling subsequent execution of the standard SVM
from scikit-learn [sklearn.svm.SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).

More detailed description of LUPI and knowledge transfer can be found in [1], [2], while specific algorithms and their options are presented in [3], [4], [5].




## 2. Data Types Supported

  The training data set should have both “standard” and  "privileged" features while the test data should only have “standard" features.
  The values of the features in dataset have to be numerical, the class labels have to be
  categorical numbers, and the missing values have to be imputed before executing the software.


## 3. Classes

### 3.1. LupiSvmClassifier

    class lupi_svm.LupiSvmClassifier(lupiregressor, C=1.0, kernel=’rbf’, degree=3, gamma=’auto_deprecated’, coef0=0.0,
    shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, 
    decision_function_shape=’ovr’, random_state=None)

    
#### 3.1.1. Parameters:

LupiSvmClassifier has the **same** input parameters as [sklearn.svm.SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
        with **one** additional parameter "lupiregressor" for a trained Lupi regressor object to learn how to generate the privileged features from the standard features.
        

    lupiregressor: LupiRegressor

        An object of regression model obtained by training on the training dataset.


#### 3.1.2. Methods:

LupiSvmClassifier inherits all the methods of [sklearn.svm.SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).
    lupi_predict(X) is the method to perform classification on samples, which does not exist in sklearn.svm.SVC.

	fit(X, y[, sample_weight])

        Fit the SVM model according to the training data that includes both standard features and the Lupi learned privileged features.

 	lupi_predict(X)

        Perform classification on samples in the dataset X with no privileged features.

### 3.2. LupiRegressor

    class lupi_svm.LupiRegressor(regr_type = 'linear', pri_features = [], krr_param_grid={}, svc_param_grid={},
                 nJobs=1, cv=6, cv_score='accuracy', decision_feature=False)

#### 3.2.1. Parameters:

    pri_features: list

        A list of the indices of the privileged features in training dataset. There should be at least one privileged feature.
      For example, Python uses 0-based indexing,  if the features with indices 0,1,5,10 in a training dataset are privileged features,
      the pri_features input should be given as: pri_features = \[0, 1, 5, 10\]

    regr_type : str, one of {'linear', 'kernelridge'} ,  Default option is 'linear'.

       Choose the type of regression to be used to train the regression models.

    nJobs : int, optional , Default option is 1.

   	   Number of jobs to run in parallel.

    krr_param_grid : dict

        non-linear LupiRegressor parameters search grid. Dictionary with parameters names (string) as keys and lists of parameter
    settings to try as values, or a list of such dictionaries, in which case the grids spanned by each dictionary in the list are
    explored.

    svc_param_grid : dict

        LupiSvmClassifer parameters search grid. Dictionary with parameters names (string) as keys and lists of parameter
    settings to try as values, or a list of such dictionaries, in which case the grids spanned by each dictionary in the list are explored.

#### 3.2.2. Methods:

	fit(X, y)

        Fit the regression model according to the training dataset which has both standard features and privileged features.

 	transform(X)

     	Transform data with only standard features to data with standard and generated (reconstructed) privileged features.

### 3.3. Composer

 	class lupi_svm.Composer(pri_features = [],
                 krr_param_grid={'gamma': [8., 2., 1.0, 0.8, 0.2], 'alpha':[0.2, 0.5, 1., 2.,4., 8., 12., 20.]},
                 svc_param_grid={'gamma': [8., 2., 1.0, 0.8, 0.2], 'C':[0.2, 0.5, 1., 2.,4., 8., 12., 20.]},
                 nJobs=1, cv=6, cv_score='accuracy',
                 random_state=None)

Composer class uses LupiRegressor and  LupiSvmClassifier classes.
 	It combines various models including linear regression (ln), non-linear regression (nl),
 	linear regression with additional decision function (ln_d)  and non-linear regression with additional decision function (nl_d),
 	and has an option to perform classification based on the cross validation performance. The default prediction output is the model with best cross-validation score.

#### 3.3.1. Parameters:
    pri_features: list

        A list of the indices of the privileged features in training dataset. There should be at least one privileged feature.

    krr_param_grid : dict

        non-linear LupiRegressor parameters search grid. Dictionary with parameters names (string) as keys and lists of parameter
    settings to try as values, or a list of such dictionaries, in which case the grids spanned by each dictionary in the list are
    explored.

    svc_param_grid : dict

        LupiSvmClassifer parameters search grid. Dictionary with parameters names (string) as keys and lists of parameter
    settings to try as values, or a list of such dictionaries, in which case the grids spanned by each dictionary in the list are explored.

    nJobs : int, optional , Default option is 1.
   	        Number of jobs to run in parallel.

#### 3.3.2. Methods:

	fit(X, y)

        Fit the regression model according to the training dataset that includes the standard features and the privileged features.

 	predict(X, type="cv")

        Perform classification on samples in the dataset X with no privileged features. It can output results with the algorithm specified by "type".
        The supported values of "type" are "ln", "nl", "ln_d", "nl_d" and "cv".

## 4. Installation

The installation should be on a python version 3 environment. The installation will pull in dependency packages
including scikit-learn, pandas, numpy, etc. You may need write permission on the installation environment.

```
$ git clone https://github.com/perspectalabs/lupi_svm.git
$ cd lupi_svm
$ pip  install .
```


## 5. Example

We use the UCI parkinsons dataset as an example.

The dataset can be download with the following command. Run the command from a shell command:

```
wget https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data
```

The statements in the following code blocks are python codes.

Import libraries:
```python
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sys
import argparse
import lupi_svm as lupi
```
Load the dataset from the dataset file to pandas data frame.
```python
data_set = pd.read_csv("parkinsons.data", header=0, delim_whitespace=False, sep=',')
```

Get the true label values y.  The "status" column is the label.
```python
y = data_set.loc[:, 'status']
```

To get the X, we need to clean the data a bit. We do following:
1. drop the "status" label column.
2. drop the "name" column which is not used.

```python
X = data_set.drop(['name', 'status'], axis=1)
```

Now X is the dataset with both standard and privileged features.

We take the non-linear features as privileged.
So, features with indices 16, 17, 18, 19, 20, 21 (0 index based) are the privileged features.

We have the privileged feature list:
```python
pri_features = [16, 17, 18, 19, 20, 21]
```

Now split the dataset to training and test sets. We take 0.33 for the test size for this example.
```python
X_train, X_test_all, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
```


Set the privileged features in the test set to 0 so that the privileged information is removed from the test set.
```python
X_test = X_test_all.copy()
X_test.iloc[:, pri_features] = 0
```

Set up the ranges and step sizes for lupi regressor grid search parameters
```python
regr_param_grid_dict = dict(gamma_list=[-2.0, 6.5, 0.4], C_list=[-2.5, 6.0, 0.4])
```

Set up the ranges and step sizes for svm grid search parameters
```python
svc_param_grid_dict = dict(gamma_list=[-2.0, 6.5, 0.4], C_list=[-2.5, 6.0, 0.4])
```

set up cross validation KFlod and score function

```python
cv_kfold = 6
cv_score = 'accuracy'
```

Get the krr_param_grid and svc_param_grid with mapping methods defined in lupi.App
```python
app = lupi.App(svc_param_grid_dict=svc_param_grid_dict,
               regr_param_grid_dict=regr_param_grid_dict,
               cv_kflod=cv_kfold,
               cv_score=cv_score)

krr_param_grid = app.get_krr_param_grid()
svc_param_grid = app.get_svc_param_grid()
```

Create the lupi classifer

```python
lupi_clf = lupi.Composer(pri_features=pri_features,
                         krr_param_grid=krr_param_grid,
                         svc_param_grid=svc_param_grid,
                         nJobs=4,
                         cv=cv_kfold,
                         cv_score=cv_score,
                         random_state=0)
```

Train model with training set
```python
print("... model training ...")
lupi_clf.fit(X_train, y_train)
```

Predict the labels of the test set.
```python
print("... predicting on test data...")
y_pred = lupi_clf.predict(X_test, type='ln')
```

Check the accuracy performance score.

```python
score = accuracy_score(y_test, y_pred)

print("...The accuracy performance on the test data is: \n   {}.".format(score))
```



-----
The above codes are in the included script *lupi_demo.py*. You can also run the example by using the script as following.

```
$python lupi_demo.py --dataset parkinsons.data --cv_score accuracy --cv_kfold 6 --test_size 0.33 --pri_features  16 17 18 19 20 21
... Loading the dataset: parkinsons.data.
... Spliting the dataset and setup training and test dataset ...
... model training ...
... predicting on test data...
...The accuracy performance on the test data is:
   0.9538461538461539.
```

## 6.  References

 1. [V. Vapnik and R. Izmailov. Learning with intelligent teacher: Similarity control and knowledge transfer. Journal of Machine Learning Research, 16, pp. 2023-2049, 2015.](http://www.jmlr.org/papers/volume16/vapnik15b/vapnik15b.pdf)
 2. V. Vapnik and R. Izmailov. Knowledge Transfer in SVM and Neural Networks. Annals of Mathematics and Artificial Intelligence, pp. 1-17, 2017.
 3. R.Ilin, R. Izmailov, Y.Goncharov, S.Streltsov, Fusion of Privileged Features for Efficient Classifier Training, Proceedings of 19th International Conference on Information Fusion,  pp. 1-8, 2016.
 4. [R. Izmailov, B. Lindqvist, and P. Lin. Feature selection in learning using privileged information.
 In 2017 IEEE International Conference on Data Mining Workshops (ICDMW), pages 957-963, 2017.](https://drive.google.com/file/d/1x3IxtloVGHUBnaOEqe7-knEnIaiooFvc/view)
 5. [R. Izmailov, P. Lin, and C. Basu. Automatic Feature Selection in Learning Using Privileged Information, 
 Proceedings of AutoML Workshop at ICML/IJCAI-ECAI, 2018. ](https://docs.google.com/viewer?a=v&pid=sites&srcid=ZGVmYXVsdGRvbWFpbnxhdXRvbWwyMDE4aWNtbHxneDo1OWNhN2FkMDMwN2RhZTM1) 
 6. [Parkinsons Data Set](https://archive.ics.uci.edu/ml/datasets/parkinsons): predict Health status of the subject
