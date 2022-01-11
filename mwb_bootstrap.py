# 
# Simply Python functions common to running Gradient Boosting predictions with SHAP analysis
#
import numpy as np
import pandas as pd
import random
import shap
from sklearn.metrics import accuracy_score, confusion_matrix, \
                            recall_score, roc_auc_score, \
                            precision_score, f1_score, matthews_corrcoef, \
                            precision_recall_curve, auc
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split


# Calculate common statistics and return a Series of them
def calc_stats(y_test, y_pred, X_test, clf, conf_flag=False):
    probs = clf.predict_proba(X_test)
    prob1 = probs[:, 1]
    stats_s = pd.Series()
    stats_s['recall'] = recall_score(y_test, y_pred)
    stats_s['prec'] = precision_score(y_test, y_pred)
    stats_s['MCC'] = matthews_corrcoef(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, prob1, pos_label=1)
    stats_s['PR_AUC'] = auc(recall, precision)
    stats_s['roc_auc'] = roc_auc_score(y_test, prob1)
    if conf_flag:
        tn, fp, fn, tp = np.around(confusion_matrix(y_test, y_pred, normalize='all').ravel()*10000,0).astype(int)
        stats_s['NTN'] = tn
        stats_s['NFP'] = fp
        stats_s['NFN'] = fn
        stats_s['NTP'] = tp

    return stats_s


def sample_data(X, y, samp_type, samp_strat, seed=0):
    if samp_type == 'over':
        sampler = RandomOverSampler(sampling_strategy=samp_strat, random_state=seed)
    elif samp_type == 'under':
        sampler = RandomUnderSampler(sampling_strategy=samp_strat, random_state=seed)
    else:
        print("Invalid 'samp_type'")
        
    # fit and apply the transform
    X_res, y_res = sampler.fit_resample(X, y)
    
    return X_res, y_res


# Run the classifier multiple time to help eliminate random variations
def bootstrap_stat(X, y, clf, nsamples=10, test_size=0.3, sample_weights=False,
                   under=False, samp_strat=1.0):
    stats_df = pd.DataFrame()
    feat_imps_df = pd.DataFrame()
    for seed in range(nsamples):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y,
                                                            random_state=seed)
        if under:
            # Undersample the training data
            X_res, y_res = sample_data(X_train, y_train, "under", samp_strat=samp_strat,
                                       seed=seed)
        else:
            X_res, y_res = X_train, y_train # Not subsampled; use with class_weight='balanced'
            #     or sample_weights
        if sample_weights:
            weights = class_weight.compute_sample_weight('balanced', y=y_res)
            clf.fit(X_res, y_res, sample_weight=weights)
        else:
            clf.fit(X_res, y_res)

        y_pred = clf.predict(X_test)

        stats_s = calc_stats(y_test, y_pred, X_test, clf)
        if stats_df.empty:
            stats_df = pd.DataFrame(stats_s)
            stats_df = stats_df.T
        else:
            stats_df = stats_df.append(stats_s, ignore_index=True)

        if feat_imps_df.empty:
            feat_imps_df = pd.DataFrame(data=clf.feature_importances_,
                                        index=X_test.columns.values, columns=[seed])
        else:
            temp_df = pd.DataFrame(data=clf.feature_importances_, index=X_test.columns.values,
                                   columns=[seed])
            feat_imps_df = feat_imps_df.merge(temp_df, left_index=True, right_index=True,
                                              how="left")

    return stats_df, feat_imps_df, X_res


# Run the classifier multiple times to help eliminate random variations
# This version includes using a smaller subsample specified by samp_size
def bootstrap_stat_samp(X, y, clf, nsamples=10, samp_size=None, test_size=0.3, sample_weights=False,
                   under=False, samp_strat=1.0):
    stats_df = pd.DataFrame()
    feat_imps_df = pd.DataFrame()
    samp_rate = []
    for seed in range(nsamples):
        if samp_size:
#            random.seed(seed)
            mask = random.sample(range(len(X)), samp_size)
            X_samp = X.iloc[mask]
#            print(X_samp.index[0])
            y_samp = y[mask]
            y_count = np.bincount(y_samp)
#            print(y_count)
            samp_rate.append(y_count[1]/len(y_samp))
#            print(f'samp_rate = {samp_rate}')
        else:
            print("Error")  # Need to fix, maybe
            return

        X_train, X_test, y_train, y_test = train_test_split(X_samp, y_samp, test_size=0.3, stratify=y_samp,
                                                            random_state=seed)
        if under:
            # Undersample the training data
            X_res, y_res = sample_data(X_train, y_train, "under", samp_strat=samp_strat,
                                       seed=seed)
        else:
            X_res, y_res = X_train, y_train # Not subsampled; use with class_weight='balanced'
            #     or sample_weights
        if sample_weights:
            weights = class_weight.compute_sample_weight('balanced', y=y_res)
            clf.fit(X_res, y_res, sample_weight=weights)
        else:
            clf.fit(X_res, y_res)

        y_pred = clf.predict(X_test)

        stats_s = calc_stats(y_test, y_pred, X_test, clf)
        if stats_df.empty:
            stats_df = pd.DataFrame(stats_s)
            stats_df = stats_df.T
        else:
            stats_df = stats_df.append(stats_s, ignore_index=True)

        if feat_imps_df.empty:
            feat_imps_df = pd.DataFrame(data=clf.feature_importances_,
                                        index=X_test.columns.values, columns=[seed])
        else:
            temp_df = pd.DataFrame(data=clf.feature_importances_, index=X_test.columns.values,
                                   columns=[seed])
            feat_imps_df = feat_imps_df.merge(temp_df, left_index=True, right_index=True,
                                              how="left")

    print(f'(no random(seed) call) Mean y_pos = {np.mean(samp_rate)}')
    return stats_df, feat_imps_df, X_res



# Run the classifier multiple times to help eliminate random variations
def bootstrap_stat_all(X, y, clf, nsamples=10, test_size=0.3, sample_weights=False, under=False, samp_strat=1.0):
    stats_df = pd.DataFrame()
    feat_imps_df = pd.DataFrame()
    y_pred = 0
    for seed in range(nsamples):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y,
                                                            random_state=seed)
        if under:
            # Undersample the training data
            X_res, y_res = sample_data(X_train, y_train, "under", samp_strat=samp_strat,
                                       seed=seed)
        else:
            X_res, y_res = X_train, y_train # Not subsampled; use with class_weight='balanced'
            #     or sample_weights
        if sample_weights:
            weights = class_weight.compute_sample_weight('balanced', y=y_res)
            clf.fit(X_res, y_res, sample_weight=weights)
        else:
            clf.fit(X_res, y_res)

        y_pred = clf.predict(X_test)

        stats_s = calc_stats(y_test, y_pred, X_test, clf)
        if stats_df.empty:
            stats_df = pd.DataFrame(stats_s)
            stats_df = stats_df.T
        else:
            stats_df = stats_df.append(stats_s, ignore_index=True)

        if feat_imps_df.empty:
            feat_imps_df = pd.DataFrame(data=clf.feature_importances_,
                                        index=X_test.columns.values, columns=[seed])
        else:
            temp_df = pd.DataFrame(data=clf.feature_importances_, index=X_test.columns.values,
                                   columns=[seed])
            feat_imps_df = feat_imps_df.merge(temp_df, left_index=True, right_index=True,
                                              how="left")

    return stats_df, feat_imps_df, X_res, y_res, X_test, y_test, y_pred
