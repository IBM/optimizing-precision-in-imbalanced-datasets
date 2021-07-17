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
from sklearn.model_selection import KFold, StratifiedKFold

class DataLoader:
    def __init__(self, config, logger):
        # Initialize Dataloader Configuration
        logging.info('[DATALOADER]: Initializing Protein Dataloader')
        self.config = config
        self.dl_cfg = self.config['dataloader']

        # Initialize PRNG Seed Values
        if self.dl_cfg.enable_seed:
            random.seed(self.dl_cfg.seed)
            np.random.seed(self.dl_cfg.seed)

        # Load Dataset
        logging.info('[DATALOADER]: Loading Dataset Files')
        if self.dl_cfg.mode == 'bio':
            logging.info('[DATALOADER]: > Loading File: ' + self.dl_cfg.bio_train_path)
            self.data = pd.read_csv(self.dl_cfg.bio_train_path, delimiter='\t', header=None)
            logging.info('[DATALOADER]: > Loaded: ' + self.dl_cfg.bio_train_path + '\t' + \
                         'Data Shape: ' + str(self.data.shape))

        elif self.dl_cfg.mode == 'phy':
            logging.info('[DATALOADER]: > Loading File: ' + self.dl_cfg.phy_train_path)
            self.data = pd.read_csv(self.dl_cfg.phy_train_path, delimiter='\t', header=None)
            logging.info('[DATALOADER]: > Loaded: ' + self.dl_cfg.phy_train_path + '\t' + \
                         'Data Shape: ' + str(self.data.shape))

    def get_data(self):
        # Initialize Crossfold Validation
        if self.dl_cfg.crossval.stratified:
            kf = StratifiedKFold(n_splits=self.dl_cfg.crossval.folds,
                                 shuffle=self.dl_cfg.crossval.shuffle,
                                 random_state=self.dl_cfg.crossval.random_state)
        else:
            kf = KFold(n_splits=self.dl_cfg.crossval.folds,
                       shuffle=self.dl_cfg.crossval.shuffle,
                       random_state=self.dl_cfg.crossval.random_state)

        for fold, (train_index, test_index) in enumerate(kf.split(self.data)):
            # Initialize Data Features and Labels
            if self.dl_cfg.mode == 'bio':
                X_train = self.data.to_numpy()[train_index, 3:]
                y_train = self.data.to_numpy()[train_index, 2].astype(int)
                X_test = self.data.to_numpy()[test_index, 3:]
                y_test = self.data.to_numpy()[test_index, 2].astype(int)
            elif self.dl_cfg.mode == 'phy':
                X_train = self.data.to_numpy()[train_index, 2:]
                y_train = self.data.to_numpy()[train_index, 1].astype(int)
                X_test = self.data.to_numpy()[test_index, 2:]
                y_test = self.data.to_numpy()[test_index, 1].astype(int)

            # Set Dataloader Attributes
            self.num_class = len(np.unique(y_train))

            yield fold, X_train, y_train, X_test, y_test
