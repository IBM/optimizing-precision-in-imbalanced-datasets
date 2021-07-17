# Copyright 2021 IBM Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import numpy as np
from sklearn.metrics import auc, roc_curve, accuracy_score, confusion_matrix, \
                            classification_report, precision_score, \
                            recall_score, f1_score, matthews_corrcoef, \
                            brier_score_loss, roc_auc_score, top_k_accuracy_score, \
                            balanced_accuracy_score
from sklearn.utils import multiclass

from collections import Counter

def evaluate_binary(y_true, y_pred):
    metrics = {}

    # Generate Classification Report
    clf_report = classification_report(y_true, y_pred, output_dict=True)

    # Flatten Output Classification Report Dictionary
    for k, v in clf_report.items():
        if type(v) == dict:
            for k_, v_ in v.items():
                metrics[k.replace(' ', '_') + '_' + k_] = v_
        else:
            metrics[k] = v

    # Compute AUC Metrics
    fpr, tpr, thresh = roc_curve(y_true, y_pred)
    metrics['auc'] = auc(fpr, tpr)

    # Compute Matthews Correlation Coefficient
    metrics['matthews'] = matthews_corrcoef(y_true, y_pred)

    # Compute Confusion Matrix
    _ = confusion_matrix(y_true, y_pred).ravel()
    if len(_) == 1:
        if y_true.unique()[0] == 1:
            metrics['TN'] = 0
            metrics['FP'] = 0
            metrics['FN'] = 0
            metrics['TP'] = _[0]
        else:
            metrics['TN'] = _[0]
            metrics['FP'] = 0
            metrics['FN'] = 0
            metrics['TP'] = 0
    elif len(_) == 4:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['TN'] = tn
        metrics['FP'] = fp
        metrics['FN'] = fn
        metrics['TP'] = tp

    return metrics

def evaluate_multi(y_true, y_pred):
    metrics = {}

    # Generate Classification Report
    clf_report = classification_report(y_true, y_pred, output_dict=True)

    # Flatten Output Classification Report Dictionary
    for k, v in clf_report.items():
        if type(v) == dict:
            for k_, v_ in v.items():
                metrics[k.replace(' ', '_') + '_' + k_] = v_
        else:
            metrics[k] = v

    # Compute Matthews Correlation Coefficient
    metrics['matthews'] = matthews_corrcoef(y_true, y_pred)

    # Compute Balanced Accuracy Score
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

    return metrics
