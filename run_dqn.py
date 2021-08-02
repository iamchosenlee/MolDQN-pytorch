import torch
from agent import Agent
from agent import QEDRewardMolecule,MultiobjectiveRewardMolecule, Agent
import hyp
import math
import utils
import numpy as np
import time
from tqdm import tqdm
import wandb
import csv
import os
from hyp import MODEL_PATH, SAVE_PATH


iterations = 200000
episode_check_interval = 10
update_interval = 20 #hparams.update_frequency
save_interval =20 #save checkpoint (hparams.save_frequency)
batch_size = 128
num_updates_per_it = 1

class Trainer():
    def __init__(self, hparams, environment, agent):
        self.hyp = hparams
        self.environment = environment
        self.agent = agent
        self.num_episodes = self.hyp.num_episodes #5000
        self.learning_frequency = self.hyp.learning_frequency #4, update_params through huber loss
        self.update_frequency = self.hyp.update_frequency #20, dqn_target = dqn
        self.save_frequency = self.hyp.save_frequency #200
        self.max_steps_per_episode = self.hyp.max_steps_per_episode #40
        self.batch_size = self.hyp.batch_size #128
        self.global_step = 0 #this is the global_step! 
        self.batch_losses = []
        self.max_reward = -999
        self.best= []
        self.log = dict() # loss, mean_loss
        self.epi_log = dict() #smiles, reward, max_reward, accumulated_reward -> remove this and just make csv
        self.save_log = dict() #best_smiles, max_reward, img
        self.WANDB = self.hyp.WANDB
        self.SIGN = self.hyp.reward_sign # for binding affinity : -1
        with open(f"{SAVE_PATH}result.csv", "w") as f:
            if self.hyp.multi_obj: 
                fieldnames = ["smiles", "reward", "custom_reward"]
            else: fieldnames = ["smiles", "reward"]
            w = csv.DictWriter(f, fieldnames = fieldnames)
            w.writeheader()

    def run_training(self):
        ## the paper trained the model for 5000 episodes (=hparams.num_episodes)
        for episode in range(self.num_episodes):
            self.global_step = self._episode()
            if (episode+1) % self.learning_frequency == 0 and self.agent.replay_buffer.__len__() >= self.batch_size:
                    loss = self.agent.update_params(self.batch_size, self.hyp.gamma, self.hyp.polyak)
                    self.batch_losses.append(float(loss))
                    self.log["loss"] = float(loss)
                    self.log["mean_loss"]= np.array(self.batch_losses).mean()
                    if self.WANDB:
                        wandb.log(self.log)
                        wandb.log(self.epi_log)
            if (episode+1) % self.update_frequency == 0:
                self.agent.update_target_dqn()
                self.batch_losses = []
                
            if (episode+1) % self.save_frequency == 0:
                self.agent.save_ckpt(self.global_step, self.log["mean_loss"], MODEL_PATH)
                print("====saved ckpt====")
                #self.log["img"] = self.environment.visualize_state(self.environment.state)
#                 self.save_log["best_smiles"].append(self.best[-1][1])
                # self.save_log["max_reward"].append(self.best[-1][2])
#                   if self.WANDB:
                    # wandb.log(self.save_log)
                    # wandb.log({"img":wandb.Image(self.environment.visualize_state(self.environment.state))})

    def _episode(self):
        # should return the updated global_step
        episode_start_time = time.time()
        eps_threshold = 1.0
        self.environment.initialize()
        if self.hyp.num_bootstrap_heads:
            self.head = np.random.randint(self.hyp.num_bootstrap_heads)
        else:
            self.head = 0
        for step in range(self.max_steps_per_episode):
            self.result = self._step(eps_threshold)
            next_state, reward, done = self.result
            if reward > self.max_reward:
                self.max_reward = reward
                self.best.append((self.global_step, self.environment.state, self.max_reward * self.SIGN))

            if step == self.max_steps_per_episode -1:
                print(
                        f"--------episode {self.global_step}--------")
                print('smiles : ', self.environment.state)
                print("it took %.3f seconds\n" %(time.time()- episode_start_time))
            
                self.epi_log["smiles"] = self.environment.state
                self.epi_log["reward"] = reward * self.SIGN
                if self.hyp.multi_obj:
                    self.epi_log["custom_reward"] = self.environment.custom_reward * self.SIGN
            
            eps_threshold *= 0.99907
        with open(f"{SAVE_PATH}result.csv", "a") as f:
            w = csv.DictWriter(f, fieldnames=self.epi_log.keys())
            w.writerow(self.epi_log)
            

        self.global_step += 1

        return self.global_step


    def _step(self, eps_threshold):
        steps_left = self.max_steps_per_episode - self.environment.num_steps_taken
        # Compute a list of all possible valid actions. (Here valid_actions stores the states after taking the possible actions)
        valid_actions = list(self.environment.get_valid_actions())

        # Append each valid action to steps_left and store in observations.
        observations = np.vstack(
            [
                np.append(
                    utils.get_fingerprint(
                        act, self.hyp.fingerprint_length, self.hyp.fingerprint_radius
                    ),
                    steps_left,
                )
                for act in valid_actions
            ]
        )  # (num_actions, fingerprint_length+1)
        observations_tensor = torch.Tensor(observations)
        # Get action through epsilon-greedy policy with the following scheduler.
        # eps_threshold = hyp.epsilon_end + (hyp.epsilon_start - hyp.epsilon_end) * \
        #     math.exp(-1. * it / hyp.epsilon_decay)

        a = self.agent.get_action(observations_tensor, self.head, eps_threshold)

        # Find out the new state (we store the new state in "action" here. Bit confusing but taken from original implementation)
        action = valid_actions[a]
        # Take a step based on the action
        result = self.environment.step(action)

        action_fingerprint = np.append(
            utils.get_fingerprint(action, self.hyp.fingerprint_length, self.hyp.fingerprint_radius),
            steps_left,
        )

        next_state, reward, done = result
        # # remember the smiles generated at this step
        # self.smiles = next_state
        # self.next_state_smi = next_state
        # Compute number of steps left
        steps_left = self.max_steps_per_episode - self.environment.num_steps_taken

        # Append steps_left to the new state and store in next_state
        next_state = utils.get_fingerprint(
            next_state, self.hyp.fingerprint_length, self.hyp.fingerprint_radius
        )  # (fingerprint_length)

        # store fingerprint of the next state
        # self.fingerprint = next_state

        action_fingerprints = np.vstack(
            [
                np.append(
                    utils.get_fingerprint(
                        act, self.hyp.fingerprint_length, self.hyp.fingerprint_radius
                    ),
                    steps_left,
                )
                for act in self.environment.get_valid_actions()
            ]
        )  # (num_actions, fingerprint_length + 1)

        # Update replay buffer (state: (fingerprint_length + 1), action: _, reward: (), next_state: (num_actions, fingerprint_length + 1),
        # done: ()

        # # if sparse buffer!!!!
        # if steps_left % 3 == 0 :
            # self.agent.replay_buffer.add(
                # obs_t=action_fingerprint,  # (fingerprint_length + 1)
                # action=0,  # No use
                # reward=reward,
                # obs_tp1=action_fingerprints,  # (num_actions, fingerprint_length + 1)
                # done=float(result.terminated),
            # )
        # return result
        self.agent.replay_buffer.add(
            obs_t=action_fingerprint,  # (fingerprint_length + 1)
            action=0,  # No use
            reward=reward,
            obs_tp1=action_fingerprints,  # (num_actions, fingerprint_length + 1)
            done=float(result.terminated),
        )
        return result



            
