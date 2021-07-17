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
import pandas as pd
import xgboost as xgb
from tqdm import tqdm

from imbopt import *
from datasets import *
from utils.eval import *
from agents.base import BaseAgent
from imbopt.util.metric import Metric
from imbopt.util.objective import Objective

class Agent(BaseAgent):
    def __init__(self, config, logger):
        super(Agent, self).__init__(config, logger)

        # Enable Debugging Logger
        if self.config.agent.debug:
            logging.getLogger().setLevel(logging.DEBUG)

        # Initialize Dataloader
        self.dataloader = eval(self.config.dataloader.name).DataLoader(config, logger)

    def run(self):
        if self.config.agent.log_metrics:
            self.train_metrics = []
            self.test_metrics = []

        for i, X_train, y_train, X_test, y_test in self.dataloader.get_data():
            logging.info('[AGENT] Cross Validation Fold ' + str(i))

            # Initialize Model Parameters
            params = {}
            params['objective'] = self.config.agent.model.obj
            if self.config.agent.model.obj.split(':')[0] == 'multi':
                params['num_class'] = self.dataloader.num_class
            params['max_depth'] = self.config.agent.model.max_depth
            params['eval_metric'] = self.config.agent.model.eval_metric

            # Initialize Model Object
            model = xgb.XGBModel()
            model.set_params(**params)

            # Perform Optimization Routine by Defined Policy
            if self.config.imbopt.name == 'default':
                logging.info('[AGENT] Training Model with Default Policy (No Optimization)')
                # Train Model w/o Optiimzation
                model.fit(X_train, y_train)
            else:
                logging.info('[AGENT] Training Model with Optimization-Based Policy')
                # Initialize Objective & Objective Policy
                data = ((X_train, y_train), (X_test, y_test))
                metric_func = Metric(self.config)
                obj = Objective(self.config, self.logger, data, model, metric_func)
                opt = eval(self.config.imbopt.name).ImbalanceOptimizer(self.config, self.logger, obj)

                # Optimize Objective Function
                logging.info('[AGENT] Initializing Optimization Routine')
                weights = opt.optimize()

                # Retrain Model with Optimal Weights
                logging.info('[AGENT] Retrain Model with Optimal Weight Parameters')
                model.fit(X_train, y_train, sample_weight=weights)

            if self.config.agent.log_metrics:
                # Perform Model Evaluation on Dataset
                logging.info('[AGENT] Evaluate Model Performance Metrics')
                y_hat_train = model.predict(X_train)
                y_hat_test = model.predict(X_test)

                if self.config.agent.model.obj.split(':')[0] == 'binary':
                    # Evaluate Model Performance
                    train_metric = evaluate_binary(y_train, y_hat_train)
                    test_metric = evaluate_binary(y_test, y_hat_test)
                elif self.config.agent.model.obj.split(':')[0] == 'multi':
                    train_metric = evaluate_multi(y_train, y_hat_train)
                    test_metric = evaluate_multi(y_test, y_hat_test)

                # Prepend Dataset Prefix to Metric Name
                new_train_metric, new_test_metric = {}, {}
                for k in train_metric: new_train_metric['train_'+k] = train_metric[k]
                for k in test_metric: new_test_metric['train_'+k] = test_metric[k]

                # Append Fold Index
                train_metric['fold'] = i
                test_metric['fold'] = i

                # Append to Metric List
                self.train_metrics.append(train_metric)
                self.test_metrics.append(test_metric)

    def finalize(self):
        if self.config.agent.log_metrics:
            # Persist Metric to Output File
            pd.DataFrame(self.train_metrics).to_csv(self.logger.tmp_dir + 'train_metric.csv', index=False)
            pd.DataFrame(self.test_metrics).to_csv(self.logger.tmp_dir + 'test_metric.csv', index=False)

            # Log Artifacts
            self.logger.log_artifact(self.logger.tmp_dir + 'train_metric.csv')
            self.logger.log_artifact(self.logger.tmp_dir + 'test_metric.csv')

