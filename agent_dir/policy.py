import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNPolicy1(nn.Module):
    def __init__(self):
        super(RNNPolicy1,self).__init__()
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        # (1, 80, 80) -> (3, 40, 40)
        self.conv1 = nn.Conv2d(1, 3, kernel_size=4, stride=2, padding=1)
        # (3, 40, 40) -> (9, 20, 20)
        self.conv2 = nn.Conv2d(3, 9, kernel_size=4, stride=2, padding=1)
        # (9, 20, 20) -> (9, 10, 10)
        self.conv3 = nn.Conv2d(9, 9, kernel_size=4, stride=2, padding=1)
        # (9 * 10 * 10) -> (2)
        self.linear = nn.Linear(9 * 10 * 10, 2)

    def forward(self, observation, train=True):
        x = self.conv1(observation)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = x.view(-1)
        x = self.linear(x)
        # for action_int
        action = self.softmax(x)
        if train:
            if action[0] > np.random.uniform():
                act_int = 2
            else:
                act_int = 3
        else:
            if action[0] > 0.5:
                act_int = 2
            else:
                act_int = 3
        return act_int, x


class LinearPolicy(nn.Module):
    def __init__(self):
        super(LinearPolicy, self).__init__()
        self.linear1 = nn.Linear(80 * 80, 200)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(200, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, observation, train=True):
        x = observation.view(-1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        # for action_int
        act_p = self.sigmoid(x)
        if train:
            if act_p > np.random.uniform():
                act_int = 2
            else:
                act_int = 3
        else:
            if act_p  > 0.5:
                act_int = 2
            else:
                act_int = 3
        return act_int, act_p


class PPO(nn.Module):
    def __init__(self,
            lr=1e-4,           
            decay_rate=0.99,
        ):
        super(PPO,self).__init__()
        # def network
        self.Policy = LinearPolicy()
        self.opt = torch.optim.RMSprop(self.Policy.parameters(), lr, weight_decay=decay_rate)

    def update(self, act_p, act, reward):
        """
        act_p : [N, 1]
        action : [N, 1]
        reward : [N]
        """
        # get label
        label = act - 2 # act is 2 and 3 -> 0, 1
        # loss
        self.opt.zero_grad()
        
        print(act_p)
        print(label)
        
        if act_p.shape[1] == 1:
            reward = reward.view(-1, 1) # pytorch BCELoss format
            loss_fn = nn.BCELoss(weight=reward) # for single output
            loss = loss_fn(act_p, label)
        else:
            label = label.view(-1).long() # pytorch CrossEntropyLoss format
            loss_fn = nn.CrossEntropyLoss(reduction="none") # for mulitple output
            loss = loss_fn(act_p, label)
            loss = torch.dot(loss, reward)
        loss.backward()
        self.opt.step()

    def get_action(self, state, train=True):
        act_int, act_p = self.Policy(state, train)
        return act_int, act_p
    
    def save(self, e=0):
        torch.save(self.Policy, "./model/Linear_" + str(e) + ".model")

    def load(self, path):
        self.Policy = torch.load(path) 
        