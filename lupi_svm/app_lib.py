from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd

from sklearn import metrics

class App(object):
    def __init__(self, svc_param_grid_dict={}, regr_param_grid_dict={}):
        self.svc_param_grid_dict = svc_param_grid_dict
        self.regr_param_grid_dict = regr_param_grid_dict
        self.set_svc_param_grid()
        self.set_krr_param_grid()
        self.cases =  ['STD', 'PRIV',
                     'ln', 'ln_d',
                     'nl', 'nl_d',
                     'cv']

        # save the metrics results for each rand_num
        # fp_ratio   -  false positives: ratio of normal data that was labeled anomalous over actual normal
        # fn_ratio   -  false negative: ratio of anomalous data that was labeled normal over actual anomalous
        # ac_ratio   -  accuracy: ratio of correct labeled (both normal and anomalous) over # of samples
        self.fp_ratio = {}
        self.fn_ratio = {}
        self.ac_ratio = {}
        self.ac_err_ratio = {}
        self.f1_score = {}

    def set_svc_param_grid(self):
        C_list = self.svc_param_grid_dict['C_list']
        gamma_list = self.svc_param_grid_dict['gamma_list']
        C_range = 2.0 ** np.arange(C_list[0], C_list[1], C_list[2])
        gamma_range = 1.0 / (2 * ((2 ** np.arange(gamma_list[0], gamma_list[1], gamma_list[2])) ** 2))
        param_grid = dict(gamma=gamma_range, C=C_range)
        self.svc_param_grid = param_grid

    def set_krr_param_grid(self):
        # For KRR, paramter alpha - small positive values of alpha improve the conditioning of the
        # problem and reduce the variance of the estimates. Alpha corresponds to (2*C)^-1
        # in other linear models such as LogisticRegression or LinearSVC. If an array is passed,
        # penalties are assumed to be specific to the targets. Hence they must correspond in number.
        C_list = self.regr_param_grid_dict['C_list']
        gamma_list = self.regr_param_grid_dict['gamma_list']
        C_range = 2.0 ** np.arange(C_list[0], C_list[1], C_list[2])
        gamma_range = 1.0 / (2 * ((2 ** np.arange(gamma_list[0], gamma_list[1], gamma_list[2])) ** 2))
        alpha_range = [1. / (2. * c) for c in C_range]
        param_grid = dict(alpha=alpha_range, gamma=gamma_range)
        self.krr_param_grid = param_grid

    def get_svc_param_grid(self):
        return self.svc_param_grid

    def get_krr_param_grid(self):
        return self.krr_param_grid

    def print_runconfig(self):
        print("==== svm_param_grid ==== {}".format(self.svc_param_grid))
        print("==== krr_param_grid ===={}".format(self.krr_param_grid))

    def add_perf(self, rand_num, targs, preds, type=None):
        accuracy = metrics.accuracy_score(targs, preds)
        accuracy_error = (1. - accuracy) * 100
        self.ac_ratio[rand_num][type] = accuracy
        self.ac_err_ratio[rand_num][type] = accuracy_error
        # calculate average fp_ratio and fn_ratio
        fp_ratio, fn_ratio = self.get_fp_fn(targs, preds)
        self.fp_ratio[rand_num][type] = fp_ratio
        self.fn_ratio[rand_num][type] = fn_ratio
        self.f1_score[rand_num][type] = metrics.f1_score(targs, preds)

    def get_fp_fn(self, targs, preds ):
        # A confusion matrix C is such that C(i,j) = the number of observations known to be in group i
        # but predicted to be in group j
        # In binary classification, the count of true negatives is C(0,0), false negative is C(1,0),
        # true positives is C(1,1) and false positives is C(0,1)

        #                  pred anormal    pred normal
        # actul anormal     c00 (TN)         c01 (FP)
        #  actul noraml     c10 (FN)         c11 (TP)

        cfm = metrics.confusion_matrix(targs, preds)
        tn = cfm.item((0,0))
        fp = cfm.item((0,1))
        fn = cfm.item((1,0))
        tp = cfm.item((1,1))
        if fp < 1e-4:
            fp_ratio = 0
        else:
            fp_ratio = 100*float(fp)/(fp+tn)
        if fn < 1e-4:
            fn_ratio = 0
        else:
            fn_ratio = 100*float(fn)/(fn+tp)
        return fp_ratio, fn_ratio

    def print_results(self, final=False):
        df_ac_err_ratio = pd.DataFrame.from_dict(self.ac_err_ratio, orient='index')
        df_fp_ratio = pd.DataFrame.from_dict(self.fp_ratio, orient='index')
        df_fn_ratio = pd.DataFrame.from_dict(self.fn_ratio, orient='index')
        df_f1_score = pd.DataFrame.from_dict(self.f1_score, orient='index')
        cases_str = '            Cases        %s' % '   '.join(map(str, self.cases))
        cases_str_csv = 'Case,%s' % ','.join(map(str, self.cases))

        if final:
            print("      *********** Summary *****************")
            print("          {}".format(cases_str))
            ac_err_mean = df_ac_err_ratio.mean(axis=0)
            ac_err_std = df_ac_err_ratio.std(axis=0)
            fp_mean = df_fp_ratio.mean(axis=0)
            fp_std = df_fp_ratio.std(axis=0)
            fn_mean = df_fn_ratio.mean(axis=0)
            fn_std = df_fn_ratio.std(axis=0)
            f1_score_mean = df_f1_score.mean(axis=0)
            f1_score_std = df_f1_score.std(axis=0)

            ac_err_std_str  =  "      accuracy error std  (%) "
            ac_err_mean_str =  "      accuracy error      (%) "
            fp_std_str      =  "      fp std              (%) "
            fp_mean_str     =  "      fp                  (%) "
            fn_std_str      =  "      fn std              (%) "
            fn_mean_str     =  "      fn                  (%) "
            f1_score_std_str = "      f1_score std            "
            f1_score_mean_str ="      f1_score                "

            ac_err_mean_csv = "accuracy error"
            f1_score_mean_csv = "f1_score"

            for case in self.cases:
                ac_err_std_str    += "    {0:.2f}".format(ac_err_std[case])
                ac_err_mean_str   += "    {0:.2f}".format(ac_err_mean[case])
                fp_std_str        += "    {0:.2f}".format(fp_std[case])
                fp_mean_str       += "    {0:.2f}".format(fp_mean[case])
                fn_std_str        += "    {0:.2f}".format(fn_std[case])
                fn_mean_str       += "    {0:.2f}".format(fn_mean[case])
                f1_score_std_str  += "    {0:.4f}".format(f1_score_std[case])
                f1_score_mean_str += "    {0:.4f}".format(f1_score_mean[case])
                ac_err_mean_csv += ",{0:.2f}".format(ac_err_mean[case])
                f1_score_mean_csv += ",{0:.4f}".format(f1_score_mean[case])

            print(ac_err_mean_str)
            print(ac_err_std_str)
            print(fp_mean_str)
            print(fp_std_str)
            print(fn_mean_str)
            print(fn_std_str)
            print(f1_score_mean_str)
            print(f1_score_std_str)
            print("\n\nwhere:")
            print(" fn (false positive): ratio of normal data that was labeled anomalous over actual normal")
            print(" fp (false negative): ratio of anomalous data that was labeled normal over actual anomalous")
            print(" accuracy: ratio of correct labeled (both normal and anomalous) over total number of samples")
            print("-------------------------------------------------------")
            print("{}".format(cases_str_csv))
            print(ac_err_mean_csv)
            print(f1_score_mean_csv)
        else:
            ac_err_current = df_ac_err_ratio.iloc[-1]
            ac_err_mean = df_ac_err_ratio.mean(axis=0)
            fp_current = df_fp_ratio.iloc[-1]
            fp_mean = df_fp_ratio.mean(axis=0)
            fn_current = df_fn_ratio.iloc[-1]
            fn_mean = df_fn_ratio.mean(axis=0)
            f1_score_current = df_f1_score.iloc[-1]
            f1_score_mean = df_f1_score.mean(axis=0)

            ac_err_current_str = "Current accuracy error (%) "
            ac_err_mean_str    = "Average accuracy error (%) "
            fp_current_str     = "   Current fp (%)          "
            fp_mean_str        = "   Average fp (%)          "
            fn_current_str     = "   Current fn (%)          "
            fn_mean_str        = "   Average fn (%)          "
            f1_score_current_str= "   Current f1_score        "
            f1_score_mean_str   = "   Average f1_score        "

            for case in self.cases:
                ac_err_current_str += "    {0:.2f}".format(ac_err_current[case])
                ac_err_mean_str    += "    {0:.2f}".format(ac_err_mean[case])
                fp_current_str     += "    {0:.2f}".format(fp_current[case])
                fp_mean_str        += "    {0:.2f}".format(fp_mean[case])
                fn_current_str     += "    {0:.2f}".format(fn_current[case])
                fn_mean_str        += "    {0:.2f}".format(fn_mean[case])
                f1_score_current_str += "    {0:.4f}".format(f1_score_current[case])
                f1_score_mean_str += "    {0:.4f}".format(f1_score_mean[case])

            print("      {}".format(cases_str))
            print(ac_err_current_str)
            print(ac_err_mean_str)
            print(fp_current_str)
            print(fp_mean_str)
            print(fn_current_str)
            print(fn_mean_str)
            print(f1_score_current_str)
            print(f1_score_mean_str)



