from torch import nn
import torch

class Actor(nn.Module):
    def __init__(self, OBSERVATION_DIM, ACTION_DIM, layers = [128,128]):
        super(Actor, self).__init__()
        self.OBS = OBSERVATION_DIM
        self.ACTION = ACTION_DIM
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
        std = torch.exp(torch.clamp(x[:,self.ACTION:], -20, 2))
        return means, std
    
    def copy(self):
        temp_actor = Actor(self.OBS, self.ACTION)
        temp_actor.load_state_dict(self.state_dict())
        return temp_actor
    
    def update(self, actor, TAU):
        for params, target_params in zip(actor.parameters(), self.parameters()):
            target_params.data.copy_(TAU * params + (1 - TAU) * target_params)

class Critic(nn.Module):
    def __init__(self, OBSERVATION_DIM, ACTION_DIM, layers = [128,128]):
        super(Critic, self).__init__()
        self.OBS = OBSERVATION_DIM
        self.ACTION = ACTION_DIM
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
        temp_critic = Critic(self.OBS, self.ACTION)
        temp_critic.load_state_dict(self.state_dict())
        return temp_critic
    
    def update(self, critic, TAU):
        for params, target_params in zip(critic.parameters(), self.parameters()):
            target_params.data.copy_(TAU * params + (1 - TAU) * target_params)