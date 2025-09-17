from torch import nn
import torch
from torch.distributions.normal import Normal

class Actor(nn.Module):
    def __init__(self, OBSERVATION_DIM, ACTION_DIM, layers, action_range, tau, std_clamp_min, std_clamp_max):
        super(Actor, self).__init__()
        self.OBS = OBSERVATION_DIM
        self.ACTION = ACTION_DIM
        self.layers = layers
        self.tau = tau
        self.action_range = action_range
        self.std_clamp_min = std_clamp_min
        self.std_clamp_max = std_clamp_max

        self.start = nn.Linear(OBSERVATION_DIM, layers[0])
        self.hidden = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(0,len(layers)-1)])
        self.end = nn.Linear(layers[-1], 2 * ACTION_DIM)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.start(state)
        x = self.relu(x)
        for layer in self.hidden:
            x = layer(x)
            x = self.relu(x)
        x = self.end(x)
        means = x[:,:self.ACTION]
        std = torch.exp(torch.clamp(x[:,self.ACTION:], self.std_clamp_min, self.std_clamp_max))
        distribution = Normal(means, std)
        if torch.is_grad_enabled():
            unsqueezed_actions = distribution.rsample()
        else:
            unsqueezed_actions = distribution.sample()
        actions = self.action_range * torch.tanh(unsqueezed_actions)
        log_probs = distribution.log_prob(unsqueezed_actions)
        log_probs = log_probs.sum(dim = -1, keepdim=True)
        log_probs -= torch.log(self.action_range * (1 - actions.pow(2))+ 1e-6).sum(dim = -1, keepdim = True)
        return actions, log_probs
    
    def copy(self):
        temp_actor = Actor(self.OBS, self.ACTION, self.layers)
        temp_actor.load_state_dict(self.state_dict())
        return temp_actor
    
    def update(self, actor):
        for params, target_params in zip(actor.parameters(), self.parameters()):
            target_params.data.copy_(self.tau * params + (1 - self.tau) * target_params)

class Critic(nn.Module):
    def __init__(self, OBSERVATION_DIM, ACTION_DIM, layers, tau):
        super(Critic, self).__init__()
        self.OBS = OBSERVATION_DIM
        self.ACTION = ACTION_DIM
        self.layers = layers
        self.tau = tau

        self.start1 = nn.Linear(OBSERVATION_DIM + ACTION_DIM, layers[0])
        self.hidden1 = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(0,len(layers)-1)])
        self.end1 = nn.Linear(layers[-1], 1)

        self.start2 = nn.Linear(OBSERVATION_DIM + ACTION_DIM, layers[0])
        self.hidden2 = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(0,len(layers)-1)])
        self.end2 = nn.Linear(layers[-1], 1)

        self.relu = nn.ReLU()

    def forward(self, state, action):
        state = torch.cat((state,action), dim = -1)
        x1 = self.start1(state)
        x1 = self.relu(x1)
        for layer in self.hidden1:
            x1 = layer(x1)
            x1 = self.relu(x1)
        x1 = self.end1(x1)

        x2 = self.start2(state)
        x2 = self.relu(x2)
        for layer in self.hidden2:
            x2 = layer(x2)
            x2 = self.relu(x2)
        x2 = self.end2(x2)

        return x1, x2
    
    def copy(self):
        temp_critic = Critic(self.OBS, self.ACTION, self.layers)
        temp_critic.load_state_dict(self.state_dict())
        return temp_critic
    
    def update(self, critic):
        for params, target_params in zip(critic.parameters(), self.parameters()):
            target_params.data.copy_(self.tau * params + (1 - self.tau) * target_params)