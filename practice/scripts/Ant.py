import torch
from torch import nn, optim
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque

from Architecture import Actor, Critic

env = gym.make("Ant-v5")
ACTION_DIM = env.action_space.shape[0]
OBS_DIM = env.observation_space.shape[0]

GAMMA = 0.99
TAU = 0.005
TARGET_ALPHA = -ACTION_DIM

ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
ALPHA_LR = 1e-5

NUM_EPOCHS = 1000
BATCH_SIZE = 256
EXPERIENCE_REPLAY_LENGTH = 300000
TRAINING_START_STEP = 10000

ACTOR_HIDDEN_LAYERS = [256,256]
CRITIC_HIDDEN_LAYERS = [256,256]

mse = nn.MSELoss()

actor = Actor(OBS_DIM, ACTION_DIM, ACTOR_HIDDEN_LAYERS)
actor_optim = optim.Adam(actor.parameters(), ACTOR_LR)
target_actor = actor.copy()

critic = Critic(OBS_DIM, ACTION_DIM, CRITIC_HIDDEN_LAYERS)
critic_optim = optim.Adam(critic.parameters(), CRITIC_LR)
target_critic = critic.copy()

alpha = torch.tensor([0.2], dtype = torch.float32, requires_grad=True)
alpha_optim = optim.Adam([alpha], ALPHA_LR)

rewards_over_time = []
experience_replay = deque([], maxlen = EXPERIENCE_REPLAY_LENGTH)

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

            target_actor.update(actor, TAU)
            target_critic.update(critic, TAU)

        if terminated or truncated:
            break

    rewards_over_time.append(current_reward)

    if epoch % 10 == 0 and epoch != 0:
        print(f"Epoch {epoch}: Current Reward {rewards_over_time[-1]:.2f} Average Reward (Last 50) {np.mean(rewards_over_time[max(0,epoch-50) :]):.2f}")

torch.save(actor.state_dict(), "Ant.pt")

plt.plot(rewards_over_time, color = 'blue', label = "Rewards Over Time")
plt.plot([np.mean(rewards_over_time[max(0, epoch-50) : epoch+1]) for epoch in range(NUM_EPOCHS)], color = 'red', label = "Average Rewards")
plt.title("Ant Training Graph")
plt.xlabel("Epochs")
plt.ylabel("Reward")
plt.grid()
plt.legend()
plt.show()



