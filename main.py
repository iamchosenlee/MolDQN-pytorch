import torch
from agent import Agent
from agent import QEDRewardMolecule,newMolecule, MultiobjectiveRewardMolecule,Agent
import hyp
import math
import utils
import numpy as np
import time
from run_dqn import Trainer
import pickle
#import wandb
import json
import os
from hyp import MODEL_PATH, SAVE_PATH


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(SAVE_PATH, exist_ok=True)


environment = newMolecule(
            discount_factor=hyp.discount_factor,
            reward_type = hyp.reward_type,
            atom_types=set(hyp.atom_types),
            init_mol=hyp.start_molecule,
            allow_removal=hyp.allow_removal,
            allow_no_modification=hyp.allow_no_modification,
            allow_bonds_between_rings=hyp.allow_bonds_between_rings,
            allowed_ring_sizes=set(hyp.allowed_ring_sizes),
            max_steps=hyp.max_steps_per_episode,
            )

# environment = MultiobjectiveRewardMolecule(
        # multi_obj_weight = hyp.multi_obj_weight, 
        # discount_factor = hyp.discount_factor,
        # target_molecule = None,
        # constraint_type = hyp.constraint_type,
        # reward_type = hyp.reward_type,
        # atom_types=set(hyp.atom_types),
        # init_mol=hyp.start_molecule,
        # allow_removal=hyp.allow_removal,
        # allow_no_modification=hyp.allow_no_modification,
        # allow_bonds_between_rings=hyp.allow_bonds_between_rings,
        # allowed_ring_sizes=set(hyp.allowed_ring_sizes),
        # max_steps=hyp.max_steps_per_episode,
        # )


# DQN Inputs and Outputs:
# input: appended action (fingerprint_length + 1) .
# Output size is (1).

if hyp.num_bootstrap_heads:
    output_length = hyp.num_bootstrap_heads
else:
    output_length = 1
agent = Agent(hyp.fingerprint_length + 1, output_length, device)


trainer = Trainer(hyp, environment, agent)
trainer.run_training()
with open(SAVE_PATH+"best_list.pkl", "wb") as f:
    pickle.dump(trainer.best, f)

print(trainer.best[-10:])

print(f"========best smiles : {trainer.best[-1][1]}========")
print(f"========best {hyp.reward_type} score : {trainer.best[-1][2]}========")
img = trainer.environment.visualize_state(trainer.best[-1][1])
img.save(SAVE_PATH+"best_smiles.png")

