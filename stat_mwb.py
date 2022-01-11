# Code credit: Shaked Zychlinski:
# https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
import pandas as pd
import numpy as np
import math
import scipy.stats as ss
from collections import Counter

from imblearn.under_sampling import RandomUnderSampler, NearMiss, CondensedNearestNeighbour, \
                                    ClusterCentroids, InstanceHardnessThreshold


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))


# @TODO - don't calculate upper-right half of matrix since it is symmetric
# @TODO - consider multiprocessing
def cramers_v_df(df) -> pd.DataFrame:
    numcols = df.shape[1]
    cdf = pd.DataFrame(np.zeros((numcols, numcols)), index=df.columns, columns=df.columns)
    for col1 in df.columns:
        for col2 in df.columns:
            cv = cramers_v(df[col1], df[col2])
            cdf.loc[col1, col2] = cv
    return cdf


# Code credit: Shaked Zychlinski:
# https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
def conditional_entropy(x, y):
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x, y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0.0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y / p_xy)
    return entropy


def theils_u(x, y):
    s_xy = conditional_entropy(x, y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n / total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        # return 1  # MWB - Changing this since perfect correlation doesn't make sense
        return 0
    else:
        return (s_x - s_xy) / s_x


def theils_u_df(df) -> pd.DataFrame:
    numcols = df.shape[1]
    udf = pd.DataFrame(np.zeros((numcols, numcols)), index=df.columns, columns=df.columns)
    for col1 in df.columns:
        for col2 in df.columns:
            tu = theils_u(df[col1], df[col2])
            udf.loc[col2, col1] = tu
    return udf


# @todo - move this routine into separate module
def under_samp(X, y, sampling_strat=1.0, target=None, under_method='RAND'):
    print(f'under_method = {under_method}')
    print(f'target = {target}')
    # Shouldn't depend on this check as it isn't "undersampling"
    if under_method == 'NONE':
        return X, y

    print(f'\nIn under_samp(): X.shape = {X.shape}; y.shape = {y.shape}\n')
    if under_method == 'RAND':
        sampler = RandomUnderSampler(sampling_strategy=sampling_strat)
        X_res, y_res = sampler.fit_resample(X, y)
    elif under_method == 'CC':
        sampler = ClusterCentroids(sampling_strategy=sampling_strat)
        X_res, y_res = sampler.fit_resample(X, y)
    elif under_method == 'INH':
        sampler = InstanceHardnessThreshold(sampling_strategy=sampling_strat)
        X_res, y_res = sampler.fit_resample(X, y)
    elif under_method == 'NM1':
        sampler = NearMiss(sampling_strategy=sampling_strat, version=1, n_neighbors=3)
        X_res, y_res = sampler.fit_resample(X, y)
    elif under_method == 'NM2':
        sampler = NearMiss(sampling_strategy=sampling_strat, version=2, n_neighbors=3)
        X_res, y_res = sampler.fit_resample(X, y)
    elif under_method == 'NM3':
        sampler = NearMiss(sampling_strategy=sampling_strat, version=3,
                           n_neighbors_ver3=3)
        X_res, y_res = sampler.fit_resample(X, y)
    elif under_method == 'CNN':
        sampler = CondensedNearestNeighbour()
        X_res, y_res = sampler.fit_resample(X, y)
    else:  # Assuming cohort undersampling
        print(f'cohort = {under_method}')
        cohort = under_method
        full_df = X.copy()
        full_df.insert(0, target, y)
        #print(full_df[[target, cohort]].groupby([target, cohort]).size())
        #print(f'full_df[target].value_counts() =\n{full_df[target].value_counts()}')

        # @TODO - Remove "hardcoded" 1's and 2's
        # Create DataFrame of "majority" (target = '1') and distribution of cohort
        major_df = full_df[full_df[target] == 1]
        major_dist = major_df[cohort].value_counts(normalize=True, sort=False).sort_index()

        # Create DataFrame of "minority" (target = '2') and corresponding distribution
        minor_df = full_df[full_df[target] == 2]
        minor_dist = minor_df[cohort].value_counts(normalize=True, sort=False).sort_index()

        # Create sample distribution rate and map it against the dataset cohort
        samp_dist = minor_dist / major_dist
        cohort_dist = full_df[cohort].map(samp_dist)

        # Randomly sample the majority class based on the sampling_strat
        samp_cnt = int(len(minor_df.index)/sampling_strat)
        major_sample_df = major_df.sample(n=samp_cnt, weights=cohort_dist)
        cohort_df = pd.concat([major_sample_df, minor_df])

        # return X & y by pulling target out separately
        X_res = cohort_df.drop(target, axis=1, inplace=False)
        print(f'len(X_res.index) = {len(X_res.index)}')
        y_res = cohort_df[target]

    return X_res, y_res

