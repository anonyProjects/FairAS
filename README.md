# An Accelerated Sampling-Based Approach for Fairness Testing of Deep Recommender Systems
This is the homepage of **$S^3$** including `tool implementation` and `experiment results`.

#### Environment configuration
Before running $S^3$, please make sure you have installed various related packages, including numpy, pandas, tensorflow, sklearn and deepctr.

You can install deepctr with the following command：

```shell
pip install deepctr
```

#### Running
Please use the following command to execute $S^3$:

```shell
python run_s.py --config=./configs/config_deepfm_lastfm_auc.json
```

#### Results
Also, we put the raw data results for all experiments in `AllResult`.

#### Data
The datasets used in the paper can be downloaded from the following link.

The deep recommendation models tested are all provided by DeepCTR. https://github.com/shenweichen/deepctr.
