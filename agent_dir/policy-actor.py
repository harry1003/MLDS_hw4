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
        self.linear_p = nn.Linear(9 * 10 * 10, 2)
        self.linear_c = nn.Linear(9 * 10 * 10, 1)

    def forward(self, observation, train=True):
        x = self.conv1(observation)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = x.view(-1)
        v = self.linear_c(x)
        p = self.linear_p(p)
        # for action_int
        action = self.softmax(x)
        if action[0] > np.random.uniform():
            ac_int = 2
        else:
            ac_int = 3
        return ac_int, p, v

class Actor_critic(nn.Module):
    def __init__(self,
            lr=1e-3,           
            decay_rate=0.99,
        ):
        super(PPO,self).__init__()
        # def network
        self.Policy = RNNPolicy1()
        self.opt = torch.optim.RMSprop(self.Policy.parameters(), lr, weight_decay=decay_rate)
    
    def update(self, act_p, act, reward, value):
        """
        act_p : [N, 1]
        action : [N, 1]
        reward : [N] 
        """
        # get label
        label = act - 2 # act is 2 and 3 -> 0, 1
        # loss
        self.opt.zero_grad()
        if act_p.shape[1] == 1:
            reward = reward.view(-1, 1) # pytorch BCELoss format
            loss_fn = nn.BCELoss(weight=reward) # for single output
            loss = loss_fn(act_p, label)
        else:
            label = label.view(-1).long() # pytorch CrossEntropyLoss format
            loss_fn = nn.CrossEntropyLoss(reduction="none") # for mulitple output
            cross_entropy = loss_fn(act_p, label)
            
        r = 0
        gamma = 0.99
        gae = torch.zeros(1).float().to(device)
        for i in range(len(r_dic) - 2, -1, -1):
            if r_dic[i] != 0:
                r = r_dic[i]
            else:
                r = r * gamma
                r_dic[i] = r
            advantage = r_dic[i] - values[i]
            value_loss += 0.5 * advantage.pow(2)
            delta_t = rewards[i] + gamma * values[i+1] - values[i]
            gae = gae * gamma + delta_t
            policy_loss += -(log_probs[i] * gae *gae + 0.01 * cross_entropy[i])

        optimizer.zero_grad()
        
        (policy_loss + 0.5 * value_loss).backward()
        
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)
        
        optimizer.step()

    def get_action(self, state):
        act_int, act_p, act_v = self.Policy(state)
        return act_int, act_p, act_v
    
    def save(self, e=0):
        torch.save(self.Policy, "./model/final_" + str(e) + ".model")
        
    def load(self, path):
        self.Policy = torch.load(path) 
        