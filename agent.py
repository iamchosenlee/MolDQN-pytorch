import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli as bernoulli
import numpy as np
import torch.optim as opt
import utils
import hyp
from dqn import MolDQN
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import AllChem
from rdkit import DataStructs
from environment import Molecule
from baselines.deepq import replay_buffer
from datetime import datetime
import collections
import pdb
# import time

from Drug_Generation import prop_predict
from Drug_Generation.test import prop_model_load as chemprop_loader

now = datetime.now()
REPLAY_BUFFER_CAPACITY = hyp.replay_buffer_size
SIGN = hyp.reward_sign


ba_model, ba_scaler = chemprop_loader('./Drug_Generation/model_0/model.pt')
def BA_score(molecule, ba_model=ba_model, ba_scalar=ba_scaler):
    """returns a BA_score"""
    score = prop_predict.get_result([molecule], ba_model, ba_scaler)
    if score[0] is None:
        return 0.0
    else:
        return score[0]

# a dictionary of custom reward functions
reward_func_dict = {'BA' : BA_score}


class MultiobjectiveRewardMolecule(Molecule):
    """ The reward is defined as a scalar
    reward = weight * constraint_score + (1 - weight) *  custom_reward """
    def __init__(self, multi_obj_weight, discount_factor, target_molecule=None, constraint_type='qed', reward_type='BA', **kwargs):
        super(MultiobjectiveRewardMolecule, self).__init__(**kwargs)
        if target_molecule:
            self.target_molecule = Chem.MolFromSmiles(target_molecule)
            self.target_mol_fingerprint = self.get_fingerprint(target_molecule)
        self.weight = float(multi_obj_weight)
        self.discount_factor = discount_factor
        assert constraint_type in ['similarity', 'qed']
        self.constraint_type = constraint_type
        assert reward_type in reward_func_dict.keys() #BA
        self.reward_type = reward_type
        self.SIGN = hyp.reward_sign
        self.custom_reward = 0.0
        
    def get_fingerprint(self, molecule):
        """ Gets the morgan fingerprint of the target molecule"""
        return AllChem.GetMorganFingerprint(molecule, radius=2)

    def get_similarity(self, smiles):
        """Gets the similarity between the current molecule and the target molecule.
        Args: 
            smiles : String
        Returns: 
            The Tanimoto similarity : Float
        """
        structure = Chem.MolFromsmiles(smiles)
        if structure is None:
                return 0.0
        fingerprint_structure = self.get_fingerprint(structure)

        return DataSructs.TanimotoSimilarity(self.target_mol_fingerprint, fingerprint_structure)
    
    def _reward(self):
        """Calculates the reward of the current state.
        Returns: A tuple of target_reward value and custom_reward"""
        if self._state is None:
            return 0.0
        mol = Chem.MolFromSmiles(self._state)
        if mol is None:
            return 0.0
        if self.constraint_type == "similarity":
            constraint_score = self.get_similarity(self._state)
        elif self.constraint_type == "qed":
            constraint_score = QED.qed(mol)
        custom_reward = reward_func_dict[self.reward_type](self._state) * self.SIGN 
        reward = (constraint_score * self.weight + custom_reward * (1 - self.weight))
        self.custom_reward = custom_reward
        return reward * self.discount_factor ** (self.max_steps - self.num_steps_taken)


class newMolecule(Molecule):
    """The molecule whose reward is from hyp.reward_fn"""

    def __init__(self, discount_factor, reward_type, **kwargs):
        """Initializes the class.

    Args:
      discount_factor: Float. The discount factor. We only
        care about the molecule at the end of modification.
        In order to prevent a myopic decision, we discount
        the reward at each step by a factor of
        discount_factor ** num_steps_left,
        this encourages exploration with emphasis on long term rewards.
      **kwargs: The keyword arguments passed to the base class.
    """
        super(newMolecule, self).__init__(**kwargs)
        self.discount_factor = discount_factor
        # self.reward_fn = ba_score
        assert reward_type in reward_func_dict.keys()
        self.reward_type = reward_type
        self.SIGN = hyp.reward_sign
    def _reward(self):
        """Reward of a state.

    Returns:
      Float. reward score  of the current state.
    """
        if self._state is None:
            return 0.0
        reward = reward_func_dict[self.reward_type](self._state) * self.SIGN
       self.custom_reward = reward
        return reward * self.discount_factor ** (self.max_steps - self.num_steps_taken)


class QEDRewardMolecule(Molecule):
    """The molecule whose reward is the QED."""

    def __init__(self, discount_factor, **kwargs):
        """Initializes the class.
    Args:
      discount_factor: Float. The discount factor. We only
        care about the molecule at the end of modification.
        In order to prevent a myopic decision, we discount
        the reward at each step by a factor of
        discount_factor ** num_steps_left,
        this encourages exploration with emphasis on long term rewards.
      **kwargs: The keyword arguments passed to the base class.
    """
        super(QEDRewardMolecule, self).__init__(**kwargs)
        self.discount_factor = discount_factor


    def _reward(self):
        """Reward of a state.

    Returns:
      Float. QED of the current state.
    """
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return 0.0
        qed = QED.qed(molecule)
        return qed * self.discount_factor ** (self.max_steps - self.num_steps_taken)


class Agent(object):
    def __init__(self, input_length, output_length, device):
        self.device = device
        self.dqn, self.target_dqn = (
            MolDQN(input_length, output_length).to(self.device),
            MolDQN(input_length, output_length).to(self.device),
        )
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.target_dqn.eval()
        self.replay_buffer = replay_buffer.ReplayBuffer(REPLAY_BUFFER_CAPACITY)
        self.optimizer = getattr(opt, hyp.optimizer)(
            self.dqn.parameters(), lr=hyp.learning_rate
        )
        self.criteria = nn.SmoothL1Loss()
        self.output_length = output_length
        if hyp.num_bootstrap_heads :
            self.num_heads = hyp.num_bootstrap_heads
            assert (self.output_length == self.num_heads)
        else:
            self.num_heads = None


    def get_action(self, observations, head, epsilon_threshold):
        # observation shape = (num_actions, fingerprint_length+1)
        if np.random.uniform() < epsilon_threshold:
            action = np.random.randint(0, observations.shape[0])
        else:
            with torch.no_grad():
                q_value = self.dqn(observations.to(self.device)).cpu()
            q_value = torch.index_select(q_value, 1, torch.tensor([head]))
            action = torch.argmax(q_value).numpy()

        return action


    def update_target_dqn(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())


    def update_params(self, batch_size, gamma, polyak):
        # update target network (optimize_model)
        # sample batch of transitions
        states, _, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        q_t = torch.zeros(batch_size, self.output_length, requires_grad=False)
        v_tp1 = torch.zeros(batch_size, self.output_length, requires_grad=False)
        for i in range(batch_size):
            state = (
                torch.FloatTensor(states[i])
                .reshape(-1, hyp.fingerprint_length + 1)
                .to(self.device)
            )
            q_t[i] = self.dqn(state)
            next_state = (
                torch.FloatTensor(next_states[i])
                .reshape(-1, hyp.fingerprint_length + 1)
                .to(self.device)
            )
            v_tp1[i] = torch.max(self.target_dqn(next_state))

        rewards = torch.FloatTensor(rewards).reshape(q_t.shape[0], 1).to(self.device)
        q_t = q_t.to(self.device)
        v_tp1 = v_tp1.to(self.device)
        dones = torch.FloatTensor(dones).reshape(q_t.shape[0],1).to(self.device)
        q_tp1_masked = (1 - dones) * v_tp1
        q_t_target = rewards + gamma * q_tp1_masked
        
        if self.num_heads:
            head_mask = bernoulli(0.6).sample((1, self.num_heads)).to(self.device)
            q_t = q_t * head_mask
            q_t_target = q_t_target * head_mask


        # aka Huber loss
        q_loss = self.criteria(q_t, q_t_target)
        
        # backpropagate
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            for p, p_targ in zip(self.dqn.parameters(), self.target_dqn.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

        return q_loss


    def save_ckpt(self, iteration, loss, model_dir):
        torch.save({
            'iteration': iteration,
            'loss': loss,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': self.dqn.state_dict()}, model_dir + now.strftime("%Y%m%d-%H%M%S")+".pt")
        print("--------checkpoint at iteration {} saved--------".format(iteration))

