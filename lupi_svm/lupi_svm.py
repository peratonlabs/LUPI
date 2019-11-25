from __future__ import division

from sklearn.model_selection import  GridSearchCV
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import pickle
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.svm import SVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import StratifiedKFold
import sys


class LupiRegressor(BaseEstimator, TransformerMixin):
    def __init__(self,  regr_type = 'linear',
                 pri_features = [],
                 krr_param_grid={}, svc_param_grid={},
                 nJobs=1, cv=6, cv_score='accuracy',
                 decision_feature=False):
        self.pri_features = pri_features
        self.num_of_std = None
        self.cv = cv
        self.cv_score = cv_score
        self.regr_list = []
        self.regr_type = regr_type
        self.scaler_lupi = None
        self.nJobs = nJobs
        self.input_shape_ = None
        self.krr_param_grid = krr_param_grid
        self.svc_param_grid = svc_param_grid
        self.decision_feature = decision_feature

    def _prepare_x (self, X_train):
        """Prepare the train set by finding out and moving the privileged columns to the end"""
        pri_features = self.pri_features
        X = check_array(X_train)
        if len(pri_features) ==0 :
            pre_X = X
        else:
            X_pri = X[:, pri_features]
            X_std = np.delete(X,np.s_[pri_features], axis=1)
            self.num_of_std = X_std.shape[1]
            pre_X = np.concatenate((X_std, X_pri), axis=1)
        return pre_X

    def _decfun (self, X_train_all, y_train_all):
        """learn the decision function feature and add it to X_train as the last column"""
        cln_len = X_train_all.shape[1] + 1
        X_decfun = np.empty([0, cln_len + 1])

        skf = StratifiedKFold(n_splits=self.cv, shuffle=False, random_state=1)

        estimator = SVC(class_weight='balanced', kernel='rbf', random_state=1)
        gs_estimator = GridSearchCV(estimator, scoring=self.cv_score, cv=self.cv,
                                    n_jobs=self.nJobs, param_grid=self.svc_param_grid)


        for k, (train_index, test_index) in enumerate(skf.split(X_train_all, y_train_all)):
            X_trn, X_tst = X_train_all[train_index], X_train_all[test_index]
            y_trn, y_tst = y_train_all[train_index], y_train_all[test_index]

            scaler = StandardScaler()
            scaler.fit(X_trn)
            X_trn_scaled = scaler.transform(X_trn)
            X_tst_scaled = scaler.transform(X_tst)

            gs_estimator.fit(X_trn_scaled, y_trn)
            dfunv = gs_estimator.decision_function(X_tst_scaled)
            dfunv_reshape = np.reshape(dfunv, (-1, 1))
            test_index_reshape = np.reshape(test_index, (-1, 1))
            # add the original index to the first column and decision value to the last column
            X_tst_dfuned = np.concatenate((test_index_reshape, X_tst, dfunv_reshape), axis=1)
            X_decfun = np.concatenate((X_decfun, X_tst_dfuned), axis=0)

        X_decfun_sorted = X_decfun[X_decfun[:, 0].argsort()]
        # delete the first column
        X_train_decfun = np.delete(X_decfun_sorted, 0, 1)
        # return the unscaled training set + the decision feature
        # the decision feature is learned from scaled training data
        return X_train_decfun

    def _lupi_transfer(self, X, num_of_std):
        # make a array of size [X.shape[0], num_of_std + len(regr_list)]
        array_over_0 = X.shape[0]
        array_over_1 = num_of_std + len(self.regr_list)
        X2_over = np.empty([array_over_0, array_over_1])

        # copy the standard feature from X
        for j in np.arange(num_of_std):
            X2_over[:, j] = X[:, j]

        # the sub array with only standard features
        X_std = X[:,0:num_of_std]

        # fill X2_over with over_regression predicted values
        over_j = num_of_std
        for regr in self.regr_list:
            X2_over[:, over_j] = regr.predict(X_std)
            over_j += 1
        # The transformed array with learned features
        X2 = X2_over[:,0:over_j]
        return X2

    def _train_regr(self, X_train, y_train):
        pre_X = self._prepare_x(X_train)
        X_train = check_array(pre_X)
        self.input_shape_ = X_train.shape

        # total number of standard + privileged features
        num_of_stdpriv = X_train.shape[1]

        ## first, let's get the lupi scalar for prediction
        num_of_std = self.num_of_std
        X_stds = X_train[:, 0:num_of_std]

        scaler_lupi = StandardScaler(copy=True, with_mean=True, with_std=True)
        scaler_lupi.fit(X_stds)
        self.scaler_lupi = scaler_lupi

        if self.decision_feature:
            X_train_dec = self._decfun(X_train, y_train)

            # total number of standard + privileged features + decision feature
            num_of_stdprivdec = X_train_dec.shape[1]
            if not isinstance(X_train_dec, pd.DataFrame):
                X_train_dec = pd.DataFrame(X_train_dec)

            #convert to numpy matirx
            X_train_dec = X_train_dec.values

            # scale the standard+privileged features
            scaler_stdpriv = StandardScaler(copy=True, with_mean=True, with_std=True)
            X_stdpriv_scaled = scaler_stdpriv.fit_transform(X_train_dec[:, 0:num_of_stdpriv])

            #  the decision feature
            X_dec = X_train_dec[:, num_of_stdpriv:num_of_stdprivdec]

            # the combined scaled train data: X_stdpriv + X_dec
            X = np.concatenate((X_stdpriv_scaled, X_dec), axis=1)
        else:
            scaler_stdpriv = StandardScaler(copy=True, with_mean=True, with_std=True)
            X_stdpriv_scaled = scaler_stdpriv.fit_transform(X_train)
            X = X_stdpriv_scaled

        ## now, fit the lupi regressors
        X_stds_scaled = X_stdpriv_scaled[:, 0:num_of_std]

        supported_regr_types = ['kernelridge','linear']
        regr_type = self.regr_type
        if regr_type not in supported_regr_types:
            print("not supported regr_type{}".format(regr_type))
            sys.exit(1)

        regrs=[]
        for j in np.arange(X.shape[1]- num_of_std):
            y_true_j = X[:, num_of_std + j]

            if regr_type == 'kernelridge':
                regr = GridSearchCV(KernelRidge(kernel='rbf'),
                                    param_grid=self.krr_param_grid,
                                    scoring='r2', cv=self.cv,
                                    n_jobs = self.nJobs)
                regr.fit(X_stds_scaled, y_true_j)
                regrs.append(regr)
            else:
                regr = LinearRegression(fit_intercept=1)
                regr.fit(X_stds_scaled, y_true_j)
                regrs.append(regr)

        return regrs

    def fit(self, X_train, y_train):
        trained_regrs = self._train_regr( X_train, y_train)
        self.regr_list = trained_regrs

    def transform(self, X):
        """ Transform the data of standard and privileged features into standard and generated (reconstructed) features.
            Parameters
            ----------
            X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples which the features are in the original order. X will be pre-processed to
            pre_x which has the features re-ordered in that the first num_of_std features are standard and
            the next features are the privileged.

            Returns
            -------
            X_transformed : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples. The first num_of_std features are standard features
             and the next features are the reconstructed privileged features.
            """
        # Check is fit had been called
        check_is_fitted(self, ['input_shape_'])

        # Input validation
        pre_X = self._prepare_x(X)
        X = check_array(pre_X)

        num_of_std = self.num_of_std
        if self.num_of_std == 0:
            print("... num_of_std can't be 0! Exiting ...")
            sys.exit(1)

        X2 = self._lupi_transfer(X, num_of_std)

        return X2

    def _predict_transform(self, X):
        """ Transform data of only standard features to data of standard and and generated (reconstructed) privileged features.
            Parameters
            ----------
            X : array-like or sparse matrix of shape = [n_samples, num_of_std]
            The test or unlabeled input data. There are num_of_std standard features.
            Returns
            -------
            X_transformed : array-like or sparse matrix of shape = [n_samples, n_features]
            The input date to feed into trained model's lupi_predict function. The first num_of_std features are
            standard features and the next features are the reconstructed privileged features.
            """
        # Check is fit had been called
        check_is_fitted(self, ['input_shape_'])

        # Input validation
        X = check_array(X)

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.num_of_std:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')

        num_of_std = self.num_of_std
        if self.num_of_std == 0:
            print("... num_of_std can't be 0! Exiting ...")
            sys.exit(1)

        X2 = self._lupi_transfer(X, num_of_std)

        return X2

class LupiSvmClassifier(SVC):
    def __init__(self, lupiregressor,
                 C=1.0, kernel='rbf', degree=3, gamma='auto',
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, decision_function_shape=None,
                 random_state=None):

        super(LupiSvmClassifier, self).__init__(
            kernel=kernel, degree=degree, gamma=gamma,
            coef0=coef0, tol=tol, C=C, shrinking=shrinking,
            probability=probability, cache_size=cache_size,
            class_weight=class_weight, verbose=verbose, max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            random_state=random_state)

        self.lupiregressor = lupiregressor

    def fit(self, X, y, sample_weight=None):
        # train the SVM model on lupi transformed training data
        X, y = check_X_y(X, y)
        X_lupi_transformed = X
        return super(LupiSvmClassifier, self).fit(X_lupi_transformed, y, sample_weight=sample_weight)

    def lupi_predict(self, X): #
        lupiregr = pickle.loads(self.lupiregressor)
        X_test_scaled = lupiregr.scaler_lupi.transform(X)
        X_test_lupilearned = lupiregr._predict_transform(X_test_scaled)
        return self.predict(X_test_lupilearned)


class Composer():
    def __init__(self,
                 pri_features = [],
                 krr_param_grid={'gamma': [8., 2., 1.0, 0.8, 0.2], 'alpha':[0.2, 0.5, 1., 2.,4., 8., 12., 20.]},
                 svc_param_grid={'gamma': [8., 2., 1.0, 0.8, 0.2], 'C':[0.2, 0.5, 1., 2.,4., 8., 12., 20.]},
                 nJobs=1, cv=6, cv_score='accuracy',
                 random_state=None):
        self.pri_features = pri_features
        self.num_of_std = None
        self.cv = cv
        self.cv_score = cv_score
        self.regr_list = []
        self.scaler_lupi = None
        self.nJobs = nJobs
        self.input_shape_ = None
        self.krr_param_grid = krr_param_grid
        self.svc_param_grid = svc_param_grid
        self.random_state = random_state

    def _fit_sanity_check(self, X_train, y_train):
        pri_features = self.pri_features
        X, y = check_X_y(X_train, y_train)
        if not isinstance(pri_features, list):
            print("The given pri_features is not a list. It is {}!!".format(pri_features))
            sys.exit(1)
        if len(pri_features) == 0:
            print(" Warning: The number of privileged features given is {}!!. Standard SVM is being used for this dataset.".format(len(pri_features)))
        else:
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
        return X, y


    def _fit_estimator(self, X_train, y_train, lupiregr_ps):
        lupi_estimator = LupiSvmClassifier(lupiregressor=lupiregr_ps, class_weight='balanced',
                                           kernel='rbf', random_state=self.random_state)

        gs_estimator = GridSearchCV(lupi_estimator, verbose=False, scoring=self.cv_score,
                                    cv=self.cv, n_jobs=self.nJobs, param_grid=self.svc_param_grid)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        lupiregressor = pickle.loads(lupiregr_ps)
        X_train_lupilearned = lupiregressor.transform(X_train)
        gs_estimator.fit(X_train_lupilearned, y_train)

        return gs_estimator

    def fit(self, X_train, y_train):
        X_train_all, y_train = self._fit_sanity_check(X_train, y_train)
        gs_estimators = {}
        X_train = X_train_all.copy()

        if len(self.pri_features) == 0:
            # no privileged feature
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            self.scaler_lupi = scaler
            estimator = SVC(class_weight='balanced', kernel='rbf', random_state=self.random_state)
            gs_estimator = GridSearchCV(estimator,
                                        scoring=self.cv_score,
                                        cv=self.cv,
                                        n_jobs=self.nJobs,
                                        param_grid=self.svc_param_grid)
            gs_estimator.fit(X_train, y_train)
            gs_estimators['std_svm'] = gs_estimator
        else:
            # regr type for linear
            regr_type = 'linear'
            lupiregressor = LupiRegressor(pri_features=self.pri_features, nJobs=self.nJobs, regr_type= regr_type,
                                          krr_param_grid=self.krr_param_grid,
                                          cv=self.cv)
            lupiregressor.fit(X_train, y_train)
            lupiregr_ps = pickle.dumps(lupiregressor)
            gs_estimators['ln'] = self._fit_estimator(X_train, y_train, lupiregr_ps)

            # regr type for non-linear
            regr_type = 'kernelridge'
            lupiregressor = LupiRegressor(pri_features=self.pri_features, nJobs=self.nJobs, regr_type= regr_type,
                                          krr_param_grid=self.krr_param_grid,
                                          cv=self.cv)
            lupiregressor.fit(X_train, y_train)
            lupiregr_ps = pickle.dumps(lupiregressor)
            gs_estimators['nl'] = self._fit_estimator(X_train, y_train, lupiregr_ps)

            # regr type for linear with decision function
            regr_type = 'linear'
            lupiregressor = LupiRegressor(pri_features=self.pri_features, nJobs=self.nJobs, regr_type= regr_type,
                                          krr_param_grid=self.krr_param_grid,
                                          svc_param_grid=self.svc_param_grid,
                                          decision_feature=True,
                                          cv=self.cv)
            lupiregressor.fit(X_train, y_train)
            lupiregr_ps = pickle.dumps(lupiregressor)
            gs_estimators['ln_d'] = self._fit_estimator(X_train, y_train, lupiregr_ps)

            # regr type for non-linear with decision function
            regr_type = 'kernelridge'
            lupiregressor = LupiRegressor(pri_features=self.pri_features, nJobs=self.nJobs, regr_type= regr_type,
                                          krr_param_grid=self.krr_param_grid,
                                          svc_param_grid=self.svc_param_grid,
                                          decision_feature=True,
                                          cv=self.cv)
            lupiregressor.fit(X_train, y_train)
            lupiregr_ps = pickle.dumps(lupiregressor)
            gs_estimators['nl_d'] = self._fit_estimator(X_train, y_train, lupiregr_ps)

        self.gs_estimators =  gs_estimators

    def _predict_sanity_check(self, X_test):
        pri_features = self.pri_features
        X_test = check_array(X_test)
        if len(pri_features) > 0:
            if max(pri_features) >= X_test.shape[1]:
                print("The given pri_features is out of the range of X_test indices. It is {}!!".format(pri_features))
                sys.exit(1)
        return X_test

    def predict(self, X_test_all, type='cv'):
        X_test_all = self._predict_sanity_check(X_test_all)
        if len(self.pri_features) == 0:
            X_test = X_test_all
            X_test = self.scaler_lupi.transform(X_test)
            estimator = self.gs_estimators['std_svm']
            y_pred = estimator.predict(X_test)
        else:
            # make the test data set contains only the standard features
            X_test = np.delete(X_test_all, self.pri_features, 1)

            predicts = {}
            for regr_type, estimator in self.gs_estimators.items():
                ## predict the test set with standard features
                y_pred = estimator.best_estimator_.lupi_predict(X_test)
                best_cvscore = estimator.best_score_
                predicts[regr_type] = (y_pred, best_cvscore)

            # get the cross validation performance
            ens_results = {}
            for key, value in predicts.items():
                ens_results[key] = {'y_pred': value[0], 'best_cvscore': value[1]}

            # convert dict to df
            er_df = pd.DataFrame.from_dict(ens_results, orient='index')
            best_regr = er_df['best_cvscore'].idxmax()
            y_pred_best = er_df.loc[best_regr, ['y_pred']]
            y_pred_cv = y_pred_best[0]
            best_cvscore_cv = er_df.loc[best_regr, ['best_cvscore']]
            predicts['cv'] = (y_pred_cv, best_cvscore_cv)
            # print("best regr:{}  best_cvscore:{}".format(best_regr, best_cvscore_cv))

            valid_types = ['ln', 'nl', 'ln_d', 'nl_d', 'cv', 'all']
            if type not in valid_types:
                print("Invalid type was given. The valid type are: {}".format(valid_types))
                sys.exit(1)
            elif type == 'all':
                y_pred = predicts
            else:
                y_pred = predicts[type][0]
        return y_pred


    def std_svm(self, X_train_all, y_train, X_test_all):
        X_train_std = X_train_all.copy()
        X_test_std = X_test_all.copy()
        X_train = X_train_std.drop(X_train_std.columns[self.pri_features], axis=1, inplace=False)
        X_test = X_test_std.drop(X_test_std.columns[self.pri_features], axis=1, inplace=False)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        estimator = SVC(class_weight='balanced', kernel='rbf', random_state=self.random_state)
        gs_estimator = GridSearchCV(estimator,
                                    scoring=self.cv_score,
                                    cv=self.cv,
                                    n_jobs=self.nJobs,
                                    param_grid=self.svc_param_grid)
        gs_estimator.fit(X_train, y_train)
        y_pred = gs_estimator.predict(X_test)
        return y_pred

    def priv_svm(self, X_train_all, y_train, X_test_all):
        X_train = X_train_all.copy()
        X_test = X_test_all.copy()

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        estimator = SVC(class_weight='balanced', kernel='rbf', random_state=self.random_state)
        gs_estimator = GridSearchCV(estimator,
                                    scoring=self.cv_score,
                                    cv=self.cv,
                                    n_jobs=self.nJobs,
                                    param_grid=self.svc_param_grid)
        gs_estimator.fit(X_train, y_train)
        y_pred = gs_estimator.predict(X_test)
        return y_pred

