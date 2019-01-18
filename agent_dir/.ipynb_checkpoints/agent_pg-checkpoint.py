### visualize ###
from comet_ml import Experiment
### visualize ###

import scipy.misc
import numpy as np
import torch


from agent_dir.agent import Agent
from .policy import PPO

def prepro(o,image_size=[80, 80]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    """
    o = o[35:,:,:]
    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
    y = y.astype(np.uint8)
    resized = scipy.misc.imresize(y, image_size)
    new_img = resized.reshape(1, 1, 80, 80)
    return new_img


class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(Agent_PG,self).__init__(env)  # we can use self.env to get right env
        # def env
        print("enviroment setup")
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        # def state
        self.pre_state = None
        self.diff_state = None
        # def ppo
        self.ppo = PPO().to(self.device)
        
        if args.test_pg:
            #you can load your model here
            print('loading trained model')
            self.ppo.load("./model/RNNPolicy0_200.model")

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        self.pre_state = np.zeros((1, 1, 80, 80))
        self.diff_state = np.zeros((1, 1, 80, 80))
        

    def train(self):
        ###################
        epochs = 100000
        batch_size = 10
        gamma = 0.99
        ###################
        
        ###### load #######
        experiment = Experiment(api_key="DFqdpuCkMgoVhT4sJyOXBYRRN") #DFqdpuCkMgoVhT4sJyOXBYRRN
        ###################
        step = 0
        r_vis = 0
        
        with experiment.train():
            for e in range(epochs):
                experiment.log_current_epoch(e)
                act_dic = []
                act_p_dic = []
                r_dic = []
                
                # init
                num_game = 1
                done = self._init_env()
                
                while True:
                    if done and num_game == batch_size:
                        ### visualize ###
                        step = step + 1
                        experiment.log_metric("reward", r_vis, step=step)
                        r_vis = 0
                        #################
                        break # finish
                    elif done:
                        ### visualize ###
                        step = step + 1
                        experiment.log_metric("reward", r_vis, step=step)
                        r_vis = 0
                        #################
                        num_game = num_game + 1 
                        done = self._init_env() # new game
                    else:
                        # in the game
                        self.diff_state = torch.from_numpy(self.diff_state).float().to(self.device)
                        act, act_p = self.ppo.get_action(self.diff_state)
                        state, reward, done, _ = self.env.step(act)
                        state = prepro(state)
                        self.diff_state = state - self.pre_state
                        self.pre_state = state
                        # remember action you take, and reward
                        act_dic.append(act)
                        act_p_dic.append(act_p)
                        r_dic.append(reward)
                        ### visualize ###
                        r_vis = r_vis + reward
                        #################
                # dic -> numpy -> tensor
                act_p_dic = torch.stack(act_p_dic)
                
                act_dic = np.array(act_dic)
                act_dic = torch.from_numpy(act_dic).float().to(self.device).view(-1, 1)
                
                r_dic = np.array(r_dic)
                r_dic = torch.from_numpy(r_dic).float().to(self.device)
                
                # early stop
                if (r_dic.sum() / batch_size) >= 2:
                    self.ppo.save(step) 

                # suitable reward 
                """
                change reward like below
                [0, 0, 0, 0, 1] -> [0.99^4, 0.99^3, 0.99^2, 0.99, 1]
                """
                r = 0
                for i in range(len(r_dic) - 1, -1, -1):
                    if r_dic[i] != 0:
                        r = r_dic[i]
                    else:
                        r = r * gamma
                        r_dic[i] = r
                r_dic = (r_dic - r_dic.mean()) / (r_dic.std() + 1e-8)
                
                self.ppo.update(act_p_dic, act_dic, r_dic)
                if(e % 100 == 0 and e != 0):
                    self.ppo.save(e) 
                    print("save:", e) 

    # set up env and pre_state, diff_state        
    def _init_env(self):
        state = self.env.reset()
        state, reward, done, info = self.env.step(0)
        state = prepro(state)
        self.pre_state = state
        state, reward, done, info = self.env.step(0)
        state = prepro(state)
        self.diff_state = state - self.pre_state
        self.pre_state = state
        return done

    def make_action(self, observation, test=True):
        state = prepro(observation)
        self.diff_state = state - self.pre_state
        self.pre_state = state
        self.diff_state = torch.from_numpy(self.diff_state).float().to(self.device)
        action, _ = self.ppo.get_action(self.diff_state, train=False)
        return action

