Preliminary code release for the CoRL 2019 paper: [HRL4IN: Hierarchical Reinforcement Learning for Interactive Navigation with Mobile Manipulators](https://arxiv.org/abs/1910.11432)

The PPO implementation is partially adopted from [pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) and [habitat-api](https://github.com/facebookresearch/habitat-api)

### Installation
1. Install [GibsonEnvV2](https://github.com/StanfordVL/GibsonEnvV2) 

2. Install HRL4IN
```
pip install -e .
```

### Usage

Train
```
./run_train.sh
```

Eval
```
./run_eval.sh
```

