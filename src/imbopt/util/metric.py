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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Metric:
    def __init__(self, config):
        # Initialize Configuration
        self.config = config
        self.metcfg = self.config.imbopt.metric

        # Initialize Metric Function
        self.metric = eval('self.' + self.metcfg.name)

    def __call__(self, y_true, y_hat):
        return self.metric(y_true, y_hat)

    def accuracy(self, y_true, y_hat):
        return accuracy_score(y_true, y_hat)

    def precision(self, y_true, y_hat):
        average = self.metcfg.params.average
        return precision_score(y_true, y_hat, average=average, zero_division=0)

    def recall(self, y_true, y_hat):
        average = self.metcfg.params.average
        return recall_score(y_true, y_hat, average=average, zero_division=0)

    def f1_score(self, y_true, y_hat):
        average = self.metcfg.params.average
        return f1_score(y_true, y_hat, average=average, zero_division=0)

    def prec_rec(self, y_true, y_hat):
        # Initialize Metric Parameters
        q_0 = self.metcfg.params.q_0
        eta = self.metcfg.params.eta
        rho = self.metcfg.params.rho
        average = self.metcfg.params.average

        # Compute Precision & Recall Metrics
        pre = precision_score(y_true, y_hat, average=average, zero_division=0)
        rec = recall_score(y_true, y_hat, average=average, zero_division=0)

        return pre - eta * max(0, q_0 * pre - rec) + rho + rec
