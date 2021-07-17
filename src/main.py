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
import hydra
import logging
from agents import *
from logger import *
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path='../conf/', config_name='../conf/config.yaml')
def main(config) -> None:
    logging.info('Initializing Butane Runtime Engine')

    # Display Configuration
    print('[Runtime Configuration]')
    print(OmegaConf.to_yaml(config))

    # Initialize Logger
    logging.info('Initialize Logger Object: ' + config.logger.name)
    logger = eval(config.logger.name).Logger(config)

    # Generate and Initialize Agent
    agent = eval(config.agent.name).Agent(config, logger)
    agent.run()
    agent.finalize()

if __name__ == '__main__':
    main()
