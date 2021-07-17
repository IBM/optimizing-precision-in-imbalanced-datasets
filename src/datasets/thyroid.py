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
        logging.info('[DATALOADER]: Initializing Thyroid Dataloader')
        self.config = config
        self.dl_cfg = self.config['dataloader']

        # Initialize PRNG Seed Values
        if self.dl_cfg.enable_seed:
            random.seed(self.dl_cfg.seed)
            np.random.seed(self.dl_cfg.seed)

        # Load Dataset
        logging.info('[DATALOADER]: Loading Dataset Files')
        data_path = eval('self.dl_cfg.'+self.dl_cfg.mode+'_path')

        # Load Training Data
        logging.info('[DATALOADER]: > Loading File: ' + data_path + '.data')
        self.train_data = self.load_data(data_path + '.data')
        logging.info('[DATALOADER]: > Loaded: ' + data_path + '.data\t' + \
                     'Data Shape: ' + str(self.train_data.shape))

        # Load Testing Data
        logging.info('[DATALOADER]: > Loading File: ' + data_path + '.test')
        self.test_data = self.load_data(data_path + '.test')
        logging.info('[DATALOADER]: > Loaded: ' + data_path + '.test\t' + \
                     'Data Shape: ' + str(self.test_data.shape))

    def load_data(self, filepath):
        # Load Raw Dataset
        raw_data = open(filepath, 'r').read().split('\n')[:-1]

        # Define Referral Source Mapping
        ref_map = {'WEST':0, 'STMW':1, 'SVHC':2, 'SVI':3, 'SVHD':4, 'other':5}

        # Perform Parsing and Data Encoding
        data = []
        for x in raw_data:
            x = x.split('|')[0]
            data.append(x.split(','))
        data = pd.DataFrame(data)

        # Perform Preprocessing
        data = data.replace(to_replace =['?'], value = np.nan)
        data = data.replace(to_replace =['M'], value = 0)
        data = data.replace(to_replace =['F'], value = 1)
        data = data.replace(to_replace =['f'], value = 0)
        data = data.replace(to_replace =['t'], value = 1)
        data.iloc[:, 0] = data.iloc[:, 0].replace(to_replace =[np.nan], value = -1).astype(int)
        data.iloc[:, 29] = data.iloc[:, 29].astype('category').cat.codes
        data.iloc[:, 28] = data.iloc[:, 28].apply(lambda x: ref_map[x])

        return data

    def get_data(self):
        # Cross Validation Fold (Since there isn't one, default at 0)
        fold = 0

        # Initialize Data Features and Labels
        X_train = self.train_data.iloc[:, :-1].to_numpy()
        y_train = self.train_data.iloc[:, -1].to_numpy()
        X_test = self.test_data.iloc[:, :-1].to_numpy()
        y_test = self.test_data.iloc[:, -1].to_numpy()

        # Set Dataloader Attributes
        self.num_class = len(np.unique(y_train))

        yield fold, X_train, y_train, X_test, y_test
