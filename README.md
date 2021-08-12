Code release for the CoRL 2019 paper: [HRL4IN: Hierarchical Reinforcement Learning for Interactive Navigation with Mobile Manipulators](https://arxiv.org/abs/1910.11432)

The PPO implementation is partially adopted from [pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) and [habitat-api](https://github.com/facebookresearch/habitat-api)

### Dependency
* torch==1.1.0
* torchvision==0.2.2

### Installation
1. Install [iGibson](https://github.com/StanfordVL/iGibson) with the `archive/hrl4in` tag.
```
cd $HOME
git clone https://github.com/StanfordVL/iGibson.git
cd $HOME/iGibson
git checkout archive/hrl4in
git submodule init
git submodule update
pip install -e .
```

2. Download [iGibson assets](https://storage.googleapis.com/gibson_scenes/assets_igibson.tar.gz).
```
wget https://storage.googleapis.com/gibson_scenes/assets_igibson.tar.gz -O /tmp/assets_igibson.tar.gz
tar -zxf /tmp/assets_igibson.tar.gz --directory $HOME/iGibson/gibson2
rm /tmp/assets_igibson.tar.gz
```

3. Install HRL4IN
```
cd $HOME
git clone --recursive https://github.com/ChengshuLi/HRL4IN.git 
cd $HOME/HRL4IN
pip install -e .
```

4. Copy the updated JR URDF file from this repo to iGibson's asset folder
```
cp $HOME/HRL4IN/hrl4in/envs/gibson/jr2_kinova.urdf $HOME/iGibson/gibson2/assets/models/jr2_urdf/jr2_kinova.urdf
```

### Usage

Train in ToyEnv
```
./run_train_toy_env.sh
```

Train in iGibson
```
./run_train_gibson.sh
```

Eval in ToyEnv
```
./run_eval_toy_env.sh
```

Eval in iGibson
```
./run_eval_gibson.sh
```

### Citation
If you use this code, please cite it as:
```
@article{li2019hrl4in,
  title={HRL4IN: Hierarchical Reinforcement Learning for Interactive Navigation with Mobile Manipulators},
  author={Li, Chengshu and Xia, Fei and Martin-Martin, Roberto and Savarese, Silvio},
  journal={arXiv preprint arXiv:1910.11432},
  year={2019}
}
```
