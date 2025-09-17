import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import gymnasium as gym
import random
from torch.distributions.normal import Normal

from Architecture import Actor, Critic

env = gym.make("Pendulum-v1")
ACTION_SPACE = env.action_space.shape[0]
OBS_SPACE = env.observation_space.shape[0]

GAMMA = 0.99
TAU = 0.005
ALPHA = 0.01

ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
ALPHA_LR = 1e-5

NUM_EPOCHS = 500
BATCH_SIZE = 256
EXPERIENCE_REPLAY_LENGTH = 10000
TRAINING_START_STEP = 2500

experience_replay = deque([], maxlen = EXPERIENCE_REPLAY_LENGTH)

rewards_over_time = []

mse = nn.MSELoss()

actor = Actor(OBS_SPACE, ACTION_SPACE, [64,64])
actor_optim = optim.Adam(actor.parameters(), ACTOR_LR)
target_actor = Actor(OBS_SPACE, ACTION_SPACE, [64,64])
target_actor.load_state_dict(actor.state_dict())

critic = Critic(OBS_SPACE, ACTION_SPACE, [64,64])
critic_optim = optim.Adam(critic.parameters(),CRITIC_LR)
target_critic = Critic(OBS_SPACE, ACTION_SPACE, [64,64])
target_critic.load_state_dict(critic.state_dict())

alpha = torch.randn((1), requires_grad=True)
alpha_optim = optim.Adam([alpha], ALPHA_LR)

for epoch in range(NUM_EPOCHS):
    state, _ = env.reset()
    state = torch.tensor(state, dtype = torch.float32)

    reward_during_epoch = 0

    while True:
        with torch.no_grad():
            action = actor(state)

            means = action[:ACTION_SPACE]
            stds = torch.exp(action[ACTION_SPACE:])

            distribution = Normal(means, stds)
            action = distribution.sample()

            next_state, reward, terminated, truncated, _ = env.step(np.array(action))

            next_state = torch.tensor(next_state, dtype = torch.float32)
            experience_replay.append((state,action,reward,next_state))
            state = next_state
            reward_during_epoch += reward
        
        if len(experience_replay) >= TRAINING_START_STEP:
            if len(experience_replay) == TRAINING_START_STEP:
                print("TRAINING STARTED")

            random_experience_array = random.sample(experience_replay, BATCH_SIZE)
            state_array, action_array, reward_array, next_state_array = zip(*random_experience_array)

            state_array = torch.stack(state_array)
            action_array = torch.stack(action_array)
            reward_array = torch.tensor(reward_array, dtype = torch.float32)
            next_state_array = torch.stack(next_state_array)
            
            with torch.no_grad():
                best_next_actions = target_actor(next_state_array)
                means = best_next_actions[:,:ACTION_SPACE]
                stds = torch.exp(best_next_actions[:,ACTION_SPACE:])
                distribution = Normal(means, stds)
                best_next_actions = distribution.sample()

                target_values = reward_array.unsqueeze(1) + GAMMA * torch.minimum(*target_critic(next_state_array, best_next_actions))
            predicted_values = critic(state_array, action_array)

            critic_optim.zero_grad()
            critic_loss = mse(target_values, predicted_values[0]) + mse(target_values, predicted_values[1])
            critic_loss.backward()
            critic_optim.step()

            best_actions = actor(state_array)
            distribution = Normal(best_actions[:,:ACTION_SPACE], torch.exp(best_actions[:,ACTION_SPACE:]))
            best_actions = distribution.rsample()
            entropy = distribution.entropy()

            alpha_optim.zero_grad()
            alpha_loss = (-alpha * (distribution.log_prob(best_actions).sum(dim = -1).detach() + ALPHA)).mean()
            alpha_loss.backward()
            alpha_optim.step()

            actor_optim.zero_grad()
            actor_loss = (-torch.minimum(*critic(state_array, best_actions)) + alpha * entropy).mean()
            actor_loss.backward()
            actor_optim.step()

            for params, target_params in zip(actor.parameters(), target_actor.parameters()):
                target_params.data.copy_(TAU * params.data + (1 - TAU) * target_params.data)
            for params, target_params in zip(critic.parameters(), target_critic.parameters()):
                target_params.data.copy_(TAU * params.data + (1 - TAU) * target_params.data)

        if terminated or truncated:
            break
    
    rewards_over_time.append(reward_during_epoch)

    if epoch % 10 == 0 and epoch != 0:
        print(f"Epoch {epoch} : Current Reward {rewards_over_time[-1]:.2f}, Average Reward (Last 50) {np.mean(rewards_over_time[max(0, epoch-50) : ]):.2f}, Alpha {alpha[0]:.2f}")

torch.save(actor.state_dict(), "Pendulum.pt")

plt.plot(rewards_over_time, color = 'blue', label = "Rewards Over Time")
plt.plot([np.mean(rewards_over_time[max(0, epoch-50) : epoch+1]) for epoch in range(NUM_EPOCHS)], color = 'red', label = "Average Rewards")
plt.title("Pendulum Training Graph")
plt.xlabel("Epochs")
plt.ylabel("Reward")
plt.grid()
plt.legend()
plt.show()
            


