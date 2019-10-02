#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import json
import math
import random
import hdbscan
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Identify outliers by using HDBSCAN
def hdbscan_outlier(X, threshold, min_cluster_size=2, **kwargs):
    X = StandardScaler().fit_transform(X)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit(X)
    outliers = np.where(clusterer.outlier_scores_ > threshold)[0]
    return outliers, clusterer.outlier_scores_

# Identify outliers using the IsolationForest method
def forest_outlier(X, threshold, behaviour='new', max_samples='auto', random_state=1234, contamination='auto', **kwargs):
    X = StandardScaler().fit_transform(X)

    clf = IsolationForest(behaviour=behaviour, max_samples=max_samples,
                        random_state=random_state, contamination=contamination)
    clf.fit(X)
    outlier_scores = clf.decision_function(X)

    outliers = np.where(outlier_scores <= threshold)[0]
    return outliers, outlier_scores

# Given a dict of resF numpy matrices, identify root_ids that lie outside the threshold using some outlier algorithm
def detect_outliers(X_c, thresholds=-0.15, outlier_method='IsolationForest', **kwargs):
    outliers = {}
    outlier_scores = {}
    for c in X_c:
        X = X_c[c]
        if isinstance(thresholds, dict):
            threshold = thresholds[c]
        else:
            threshold = thresholds
        
        if outlier_method == 'IsolationForest':
            outliers_c, outlier_scores_c = forest_outlier(X, threshold, **kwargs)
        elif outlier_method == 'HDBSCAN':
            outliers_c, outlier_scores_c = hdbscan_outlier(X, threshold, **kwargs)

        outliers[c] = outliers_c
        outlier_scores[c] = outlier_scores_c
    return outliers, outlier_scores

# Given a resF numpy matrix, identify root_ids within the matrix that lie outside the threshold using some outlier algorithm
def outlier_update(X, threshold=-0.15, outlier_method='IsolationForest', **kwargs):    
    if outlier_method == 'IsolationForest':
        outliers, outlier_scores = forest_outlier(X, threshold, **kwargs)
    elif outlier_method == 'HDBSCAN':
        outliers, outlier_scores = hdbscan_outlier(X, threshold, **kwargs)

    return outliers
