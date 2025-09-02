from torch import nn
import torch

class Actor(nn.Module):
    def __init__(self, OBSERVATION_DIM, ACTION_DIM, layers = [128,128]):
        super(Actor, self).__init__()
        self.start = nn.Linear(OBSERVATION_DIM, layers[0])
        self.hidden = [nn.Linear(layers[i], layers[i+1]) for i in range(1,len(layers)-1)]
        self.end = nn.Linear(layers[-1], 2 * ACTION_DIM)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.start(state)
        x = self.relu(x)
        for layer in self.hidden:
            x = layer(x)
            x = self.relu(x)
        x = self.end(x)
        return x

class Critic(nn.Module):
    def __init__(self, OBSERVATION_DIM, ACTION_DIM, layers = [128,128]):
        super(Critic, self).__init__()
        self.start1 = nn.Linear(OBSERVATION_DIM + ACTION_DIM, layers[0])
        self.hidden1 = [nn.Linear(layers[i], layers[i+1]) for i in range(1,len(layers)-1)]
        self.end1 = nn.Linear(layers[-1], 1)

        self.start2 = nn.Linear(OBSERVATION_DIM + ACTION_DIM, layers[0])
        self.hidden2 = [nn.Linear(layers[i], layers[i+1]) for i in range(1,len(layers)-1)]
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