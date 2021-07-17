# Optimizing Precision in Imbalanced Datasets
Machine Learning Framework for Weighted Metric Optimization to Handle Imbalanced Datasets

Implementation of **"Optimizing predictive precision in imbalanced datasets for
actionable revenue change prediction"** by *Mahajan et. al*

## Setup
### Environment Setup
The environment uses Anaconda setup for setting up the dependencies required to
run this project. To setup the necessary dependencies to run experiments,
execute the following command from the root directory of the repository:
```
./setup.sh <env_name>
```
Where `<env_name>` is the name of the conda environment that will be used. Prior
to generating the environment, you can activate the conda environment by:
```
conda activate <env_name>
```

## Dataset
To train the models on the sample datasets that were used in the paper, you can
run the following dataset download script:
```
./dataset_setup.sh
```

This will download the following datasets, which are made available through the
dataloaders that have already been implemented. To use a particular dataset for
the experiment, you change the `dataloader` attribute in the configuration file
found under `conf/config.yaml`.

|    Dataset    |    Parameter    |                Mode               | Data Columns |         CV         |
|:-------------:|:---------------:|:---------------------------------:|:------------:|:------------------:|
| Optical Digit | `optical_digit` |                N/A                |      65      |         :x:        |
|    Protein    |    `protein`    |             `bio, phy`            |    77, 81    | :white_check_mark: |
|    Satelite   |    `satelite`   |                N/A                |      37      |         :x:        |
|     Scene     |     `scene`     |                N/A                |      300     | :white_check_mark: |
|  Spetrometer  |  `spetrometer`  |                N/A                |      101     | :white_check_mark: |
|    Thyroid    |    `thyroid`    | `bp, dis, hyper, hypo, rep, sick` |      30      |         :x:        |
|      Wine     |      `wine`     |                N/A                |      12      | :white_check_mark: |

Certain datasets may contain several variety of dataset modes, you can adjust
them accordingly based on whether or not `mode` is available in the
configuration file.

## Configuration
Detailed documentation regarding configuration to be updated soon!

## Citation
To cite the original paper of this work, please use the following ciation:
```
@article{MAHAJAN20201095,
    title = {Optimizing predictive precision in imbalanced datasets for actionable revenue change prediction},
    journal = {European Journal of Operational Research},
    volume = {285},
    number = {3},
    pages = {1095-1113},
    year = {2020},
    issn = {0377-2217},
    doi = {https://doi.org/10.1016/j.ejor.2020.02.036},
    url = {https://www.sciencedirect.com/science/article/pii/S0377221720301715},
    author = {Pravar Dilip Mahajan and Abhinav Maurya and Aly Megahed and Alaa Elwany and Ray Strong and Jeanette Blomberg},
    keywords = {(D) Analytics, Revenue change prediction, Classification, Machine learning, Bayesian optimization, Imbalanced datasets},
    abstract = {In business environments where an organization offers contract-based periodic services to its clients, one crucial task is to predict changes in revenues generated through different clients or specific service offerings from one time epoch to another. This is commonly known as the revenue change prediction problem. In practical real-world environments, the importance of having adequate revenue change prediction capability primarily stems from scarcity of resources (in particular, sales team personnel or technical consultants) that are needed to respond to different revenue change scenarios including predicted revenue growth or shrinkage. It becomes important to make actionable decisions; that is, decisions related to prioritizing clients or service offerings to which these scarce resources are to be allocated. The contribution of the current work is twofold. First, we propose a framework for conducting revenue change prediction through casting it as a classification problem. Second, since datasets associated with revenue change prediction are typically imbalanced, we develop a new methodology for solving the classification problem such that we achieve maximum prediction precision while minimizing sacrifice in prediction accuracy. We validate our proposed framework through real-world datasets acquired from a major global provider of cloud computing services, and benchmark its performance against standard classifiers from previous works in the literature.}
}
```

If you used any part of the code in your research, please cite our repository
using the following citation:
```
@Misc{,
    author = {Yuya Jeremy Ong},
    title = {Optimizing Precision in Imbalanced Datasets},
    year = {2021--},
    url = " https://github.com/IBM/optimizing-precision-in-imbalanced-datasets"
}
```