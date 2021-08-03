# MolDQN-pytorch
[![MIT
license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

PyTorch implementation of MolDQN as described in [Optimization of Molecules via Deep Reinforcement Learning](https://www.nature.com/articles/s41598-019-47148-x)
by Zhenpeng Zhou, Steven Kearnes, Li Li, Richard N. Zare and Patrick Riley.

Forked from https://github.com/aksub99/MolDQN-pytorch

## Added Features and Differences
* `agent.py` has additional newMolecule, MultiobjectiveREwardMolecule for single optimization / multi-objective optimization with custom reward functions other than QED.
* Bootstrap DQN option is added to follow the original paper.
* Tensorboard is removed and wandb is added for logging and visualization.
* `run_dqn.py` is added to resemble the original tensorflow implementation.

## Training the MolDQN:

`python run_dqn.py`

To use **wandb** for better visualization, run `wandb.ipynb`.

## Results:

The wandb results should look like the following image.
![image](https://user-images.githubusercontent.com/29084981/128026350-b8b1b1e2-66b0-44c8-88bb-bb06446589f7.png)


The logs should look like the following image.
![image](https://user-images.githubusercontent.com/29084981/128024644-e93f9cb3-e63c-44a6-b939-5a040b6367b3.png)


## References:
The original tensorflow implementation can be found at https://github.com/google-research/google-research/tree/master/mol_dqn
This repository re-uses some code from the original implementation.
