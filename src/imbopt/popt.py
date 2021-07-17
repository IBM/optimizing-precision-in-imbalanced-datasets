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
import logging
import operator
import numpy as np

class ImbalanceOptimizer:
    def __init__(self, config, logger, objective):
        # Initialize Core Attributes
        self.config = config
        self.logger = logger
        self.objective = objective
        self.io_cfg = self.config.imbopt

        # Initialize Weights
        # TODO: Define methods for how to initialize weights.
        self.weights = np.ones((len(self.objective.y_train), ))
        self.objective.init_weights(self.weights)

    def optimize(self):
        if self.io_cfg.verbose:
            logging.info('[POPT]: Initializing Grid Search Optimization Routine')

        # Initialize Metric and Grid Value Tracking
        metrics = {}

        # Interpolate Over Different Grid Values
        for i in range(1, self.io_cfg.grid_granularity):
            # Compute Grid Weight Value
            grid_value = 1.0 * i / self.io_cfg.grid_granularity

            # Compute Score Based on Grid Value
            metrics[grid_value] = self.objective.get_objective(grid_value)

            if self.io_cfg.verbose:
                logging.info('[POPT]: Grid Value: ' + str(grid_value) + '\t' + \
                             'Metric: ' + str(metrics[grid_value]))

        # Obtain Optimal Value with Largest Metric
        value = max(metrics.items(), key=operator.itemgetter(1))[0]

        if self.io_cfg.verbose:
            logging.info('[POPT]: Optimal Value: ' + str(value))

        self.weights = np.array([self.weights[i] * (value if (self.objective.y_train[i] == 1)
                        else (1 - value)) for i in range(len(self.weights))])

        return self.weights