import torch
from torch import nn, optim
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import sys
import yaml
import csv

from Architecture import *

hyperparams_file_name = sys.argv[1] + "/hyperparameters.yml"

with open(hyperparams_file_name, 'r') as f:
    hyperparams = yaml.safe_load(f)

env = gym.make(hyperparams['ENV_NAME'])
ACTION_DIM = env.action_space.shape[0]
OBS_DIM = env.observation_space.shape[0]
action_range = torch.tensor(env.action_space.high)

actor, target_actor, critic, target_critic = get_architecture(OBS_DIM, ACTION_DIM, hyperparams['ACTOR_HIDDEN_LAYERS'], hyperparams['CRITIC_HIDDEN_LAYERS'], action_range, hyperparams['TAU'], hyperparams['STD_CLAMP_MIN'], hyperparams['STD_CLAMP_MAX'])
actor_optim = optim.Adam(actor.parameters(), hyperparams['ACTOR_LR'])
critic_optim = optim.Adam(critic.parameters(), hyperparams['CRITIC_LR'])

GAMMA = hyperparams['GAMMA']
TARGET_ALPHA = hyperparams['TARGET_ALPHA']

alpha = torch.tensor([0.2], dtype = torch.float32, requires_grad=True)
alpha_optim = optim.Adam([alpha], hyperparams['ALPHA_LR'])

NUM_EPOCHS = hyperparams['NUM_EPOCHS']
BATCH_SIZE = hyperparams['BATCH_SIZE']
EXPERIENCE_REPLAY_LENGTH = hyperparams['EXPERIENCE_REPLAY_LENGTH']
TRAINING_START_STEP = hyperparams['TRAINING_START_STEP']

LOGGING_FREQ = hyperparams['LOGGING_FREQ']
SLIDING_WINDOW_AVERAGE = hyperparams['SLIDING_WINDOW_AVERAGE']

mse = nn.MSELoss()
rewards_over_time = []
experience_replay = deque([], maxlen = EXPERIENCE_REPLAY_LENGTH)

loss_array = []

best_reward = -1000000

for epoch in range(NUM_EPOCHS):
    state, _ = env.reset()
    state = torch.tensor(state, dtype = torch.float32).unsqueeze(0)

    current_reward = 0
    
    while True:
        with torch.no_grad():
            action, _ = actor(state)
            next_state, reward, terminated, truncated, _ = env.step(np.asarray(action[0]))
            next_state = torch.tensor(next_state, dtype = torch.float32).unsqueeze(0)
            experience_replay.append((state[0],action[0], reward, next_state[0], 1 if terminated else 0))
            state = next_state
            current_reward += reward

        if len(experience_replay) >= TRAINING_START_STEP:
            if len(experience_replay) == TRAINING_START_STEP:
                print(f"TRAINING STARTED: EPOCH {epoch} AVG RANDOM SCORE {np.mean(rewards_over_time):.2f}")
            
            random_experiences = random.sample(experience_replay, BATCH_SIZE)
            state_array, action_array, reward_array, next_state_array, done_array = zip(*random_experiences)

            state_array = torch.stack(state_array)
            action_array = torch.stack(action_array)
            reward_array = torch.tensor(reward_array, dtype = torch.float32)
            next_state_array = torch.stack(next_state_array)
            done_array = torch.tensor(done_array, dtype = torch.float32)

            with torch.no_grad():
                best_next_actions, next_log_probs = target_actor(next_state_array)
                target_state_action_value = reward_array.unsqueeze(1) + GAMMA * (torch.minimum(*target_critic(next_state_array, best_next_actions.detach())) - alpha.detach() * next_log_probs.detach()) * (1 - done_array).unsqueeze(1)
            predicted_state_action_value = critic(state_array, action_array)

            critic_optim.zero_grad()
            critic_loss = mse(target_state_action_value, predicted_state_action_value[0]) + mse(target_state_action_value, predicted_state_action_value[1])
            critic_loss.backward()
            critic_optim.step()

            best_actions, log_probs = actor(state_array)

            actor_optim.zero_grad()
            actor_loss = (-torch.minimum(*critic(state_array, best_actions)) + alpha * log_probs.sum(dim=-1)).mean()
            actor_loss.backward()
            actor_optim.step()

            alpha_optim.zero_grad()
            alpha_loss = (-alpha * (log_probs.sum(dim = -1).detach() + TARGET_ALPHA)).mean()
            alpha_loss.backward()
            alpha_optim.step()

            target_actor.update(actor)
            target_critic.update(critic)

            loss_array.append((actor_loss.detach().itme(), critic_loss.detach().item(), alpha_loss.detach().item()))

        if terminated or truncated:
            break

    rewards_over_time.append(current_reward)

    if epoch % LOGGING_FREQ == 0 and epoch != 0:
        print(f"Epoch {epoch}: Current Reward {rewards_over_time[-1]:.2f} Average Reward (Last {SLIDING_WINDOW_AVERAGE}) {np.mean(rewards_over_time[max(0,epoch-SLIDING_WINDOW_AVERAGE):]):.2f}")
        if np.mean(rewards_over_time[max(0,epoch-SLIDING_WINDOW_AVERAGE):]) > best_reward:
            best_reward = np.mean(rewards_over_time[max(0,epoch-SLIDING_WINDOW_AVERAGE):])
            torch.save(actor.state_dict(), hyperparams['FILE_NAME'] + '.pt')

with open(hyperparams['FILE_NAME'] + 'TrainingLoss.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(loss_array)

with open(hyperparams['FILE_NAME'] + 'TrainingRewards.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(rewards_over_time)

plt.plot(rewards_over_time, color = 'blue', label = "Rewards Over Time")
plt.plot([np.mean(rewards_over_time[max(0, epoch-50) : epoch+1]) for epoch in range(NUM_EPOCHS)], color = 'red', label = "Average Rewards")
plt.title("Rewards During Training")
plt.xlabel("Epochs")
plt.ylabel("Reward")
plt.grid()
plt.legend()
plt.savefig(hyperparams['FILE_NAME'] + 'RewardsGraph.png')

plt.figure()

actor_loss, critic_loss, alpha_loss = zip(*loss_array)
plt.plot(actor_loss, label = 'Actor Loss', color = 'blue')
plt.plot(critic_loss, label = 'Critic Loss', color = 'red')
plt.plot(alpha_loss, label = 'Alpha Loss', color = 'green')
plt.title("Loss During Training")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.grid()
plt.legend()
plt.savefig(hyperparams['FILE_NAME'] + 'TrainingLossGraph.png')