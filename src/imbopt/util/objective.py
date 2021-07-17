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

class Objective:
    def __init__(self, config, logger, data, model, metric, weights=None):
        # Initialize Core Attributes
        self.config = config
        self.logger = logger

        # Decompose Train/Test Data Tuple
        self.X_train, self.y_train = data[0]
        self.X_test, self.y_test = data[1]

        # Initialize Objective Object Attributes
        self.model = model
        self.metric = metric
        self.weights = weights

    def init_weights(self, weights):
        self.weights = weights

    def get_objective(self, gridValue):
        # Validate Weights
        if self.weights is None:
            raise ValueError('Optimization weights were not initialized.')

        # Return Negative Score If Class is Uniform
        if np.var(self.y_train) == 0.0: return -100

        # Update Weight Values
        w_updated = [self.weights[i] * (gridValue if (self.y_train[i] == 1)
                     else (1 - gridValue)) for i in range(len(self.weights))]

        # Train Model with Updated Weights
        self.model.fit(self.X_train, self.y_train, sample_weight=w_updated)

        # Compute Objective Metrics
        y_hat_test = self.model.predict(self.X_test)

        return self.metric(self.y_test, y_hat_test)
