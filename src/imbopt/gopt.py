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
from bayes_opt import BayesianOptimization

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

        # Initialize Optimizer
        self.bo = BayesianOptimization(self.objective.get_objective,
                    {'gridValue': (self.io_cfg.grid_value.lower,
                                   self.io_cfg.grid_value.upper)})

    def optimize(self):
        # Pre-Search Iterpolation Over Search Space
        probe_points = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999]
        for pt in probe_points: self.bo.probe(params={'gridValue': pt})
        self.bo.maximize(init_points=self.io_cfg.init_points,
                         n_iter=self.io_cfg.n_iterations,
                         acq='ucb')

        # Obtain Optimal Weight Values from Optimization
        value = self.bo.max['params']['gridValue']
        self.weights = np.array([self.weights[i] * (value if (self.objective.y_train[i] == 1)
                        else (1 - value)) for i in range(len(self.weights))])

        return self.weights
