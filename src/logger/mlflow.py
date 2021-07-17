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
import os
import sys
import uuid
import hydra
import shutil
import logging
from omegaconf import DictConfig, ListConfig

import mlflow
from mlflow.tracking.client import MlflowClient

from logger.base import BaseLogger

class Logger():
    def __init__(self, config):
        # Initialize Configuration and Directory Paths
        self.config = config
        self.out_dir = str(hydra.utils.get_original_cwd()) + '/' + config.logger.output_dir
        self.tmp_dir = str(hydra.utils.get_original_cwd()) + '/' + config.logger.temp_dir
        logging.info('MLFlow Artifact Directory: ' + self.out_dir)
        logging.info('MLFlow Temporary Directory: ' + self.tmp_dir)

        # Initialize Debug Mode - If Enabled
        if self.config.agent.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logging.debug('Debugging Mode: Enabled')

        # Initialize MLFlow Client
        mlflow.set_tracking_uri('file://' + self.out_dir)
        self.client = MlflowClient('file://' + self.out_dir)
        logging.info('Set MLFLow Client Tracking URI: ' + 'file://' + self.out_dir)

        # Initialize MLFlow Experiment
        # TODO: Figure out how to enable this runtime using MLFlow Run cli directly.
        try:
            self.experiment_id = self.client.create_experiment(self.config.exp_name)
            logging.info('Initializing New MLFlow Experiment: ' + self.config.exp_name)
        except:
            self.experiment_id = self.client.get_experiment_by_name(self.config.exp_name).experiment_id
            logging.info('Initializing MLFlow Experiment: ' + self.config.exp_name)
        logging.info('MLFlow Experiment ID: ' + str(self.experiment_id))

        # Initialize Run ID
        self.run = self.client.create_run(self.experiment_id)
        self.run_id = self.run.info.run_id
        logging.info('Initializing Run ID: ' + self.run_id)

        # Initialize Run
        mlflow.start_run(self.run_id, self.experiment_id)
        logging.info('Initializing MLFlow Run Logging')

    def __del__(self):
        # Terminate MLFlow Run
        mlflow.end_run()
        logging.info('Terminating MLFlow Run Logging')

        # Log Configuration Parameters
        self.log_params_from_omegaconf_dict(self.config)
        logging.info('Logging Configuration Parameters in MLFlow Logger')

        # Register Hydra Artifacts
        logging.info('Persisting Hydra Configuration File Artifacts')
        self.log_artifact(self.tmp_dir + '.hydra/config.yaml')
        self.log_artifact(self.tmp_dir + '.hydra/hydra.yaml')
        self.log_artifact(self.tmp_dir + '.hydra/overrides.yaml')
        self.log_artifact(self.tmp_dir + 'main.log')

        # Cleanup Temporary Directory
        shutil.rmtree(self.tmp_dir)

    def log_params_from_omegaconf_dict(self, params):
        for param_name, element in params.items():
            self._explore_recursive(param_name, element)

    def _explore_recursive(self, parent_name, element):
        if isinstance(element, DictConfig):
            for k, v in element.items():
                if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                    self._explore_recursive(f'{parent_name}.{k}', v)
                else:
                    self.client.log_param(self.run_id, f'{parent_name}.{k}', v)
        elif isinstance(element, ListConfig):
            for i, v in enumerate(element):
                self.client.log_param(self.run_id, f'{parent_name}.{i}', v)

    def log_param(self, key, value):
        self.client.log_param(self.run_id, key, value)

    def log_metric(self, key, value):
        self.client.log_metric(self.run_id, key, value)

    def log_metrics(self, metrics):
        for k, v in metrics.items():
            self.log_metric(k, v)

    def log_artifact(self, local_path):
        logging.info('[MLFlow][Artifact]: Logged ' + local_path)
        self.client.log_artifact(self.run_id, local_path)

