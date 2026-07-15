# policy-stack

A flexible reinforcement learning library built to support high-level and low-level policy training

'''bash
pip install policy-stack
'''

## Quickstart

'''python

'''

## Features

- **Algorithms**: PPO
- **Flexible rollout storage**: Handles collection and batching of any type of data with 'stackable' (invariably-dimensioned tensors), 'sequential' (variably-dimensioned tensors), and 'opaque' (non-tensors) field types
- **Configurable trainers**: Fully-configurable high-level training interfaces

## Architecture

'''
policystack/
├── reinforcement-learning/  
│   └── ppo.py               # PPO trainer; PPO-specific Actor/Critic
├── math/                    
│   ├── advantage.py         # GAE, TD, MC advantage computers
│   └── objective.py         # PPO objective
├── utils/
│   └── containers.py        # RolloutBuffer
└── config.py                # Configurations for all RL trainers
'''

## Citation
 
```bibtex
@software{policystack,
  author = {Harrison Deurwaarder},
  title = {policystack: Lightweight & Modular RL Library},
  year = {2026},
  url = {https://github.com/HarrisonDeurwaarder/policystack}
}
```