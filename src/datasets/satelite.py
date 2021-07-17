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
import random
import logging
import numpy as np
import pandas as pd

class DataLoader:
    def __init__(self, config, logger):
        # Initialize Dataloader Configuration
        logging.info('[DATALOADER]: Initializing Satelite Dataloader')
        self.config = config
        cfg = self.config['dataloader']

        # Initialize PRNG Seed Values
        if cfg.enable_seed:
            random.seed(cfg.seed)
            np.random.seed(cfg.seed)

        # Load Dataset
        logging.info('[DATALOADER]: Loading Dataset Files')
        logging.info('[DATALOADER]: > Loading File: ' + cfg.train_path)
        logging.info('[DATALOADER]: > Loading File: ' + cfg.test_path)

        self.train_data = pd.read_csv(cfg.train_path, header=None, delimiter=' ')
        self.test_data = pd.read_csv(cfg.test_path, header=None, delimiter=' ')

        logging.info('[DATALOADER]: > Loaded: ' + cfg.train_path + '\t' + \
                     'Data Shape: ' + str(self.train_data.shape))
        logging.info('[DATALOADER]: > Loaded: ' + cfg.test_path + '\t' + \
                     'Data Shape: ' + str(self.test_data.shape))

    def get_data(self):
        # Cross Validation Fold (Since there isn't one, default at 0)
        fold = 0

        # Initialize Data Features and Labels
        X_train = self.train_data.iloc[:, :-1]
        y_train = self.train_data.iloc[:, -1].replace(7, 6).to_numpy() - 1
        X_test = self.test_data.iloc[:, :-1]
        y_test = self.test_data.iloc[:, -1].replace(7, 6).to_numpy() - 1

        # Set Dataloader Attributes
        self.num_class = len(np.unique(y_train))

        yield fold, X_train, y_train, X_test, y_test
