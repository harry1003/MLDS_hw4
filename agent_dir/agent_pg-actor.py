### visualize ###
from comet_ml import Experiment
experiment = Experiment(api_key="DFqdpuCkMgoVhT4sJyOXBYRRN")
### visualize ###

import scipy.misc
import numpy as np
import torch

from agent_dir.agent import Agent
from .policy-actor import Actor_critic

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
        # def model
        self.model = Actor_critic().to(self.device)
        
        if args.test_pg:
            #you can load your model here
            print('loading trained model')

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        _ = self._init_env()


    def train(self):
        ## highper parameter ##
        epochs = 10000
        N_game_per_epoch = 10
        gamma = 0.95
        ## highper parameter ##
        
        r_vis = 0
        step = 0
        with experiment.train():
            e = 0
            num_step = 256
            while True:
                # init
                num_game = 1
                done = self._init_env()
                
                e = e + 1
                act_dic = []
                act_p_dic = []
                r_dic = []
                v_dic = []
                
                for i in range(num_step)
                    if done:
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
                        act, act_p, value = self.model.get_action(self.diff_state)
                        state, reward, done, _ = self.env.step(act)
                        state = prepro(state)
                        self.diff_state = state - self.pre_state
                        self.pre_state = state
                        # remember action you take, and reward
                        act_dic.append(act)
                        act_p_dic.append(act_p)
                        r_dic.append(reward)
                        v_dic.append(value)
                        ### visualize ###
                        r_vis = r_vis + reward
                        #################
                # dic -> numpy -> tensor
                act_p_dic = torch.stack(act_p_dic)
                
                v_dic = torch.stack(v_dic)
                
                act_dic = np.array(act_dic)
                act_dic = torch.from_numpy(act_dic).float().to(self.device).view(-1, 1)
                
                r_dic = np.array(r_dic)
                r_dic = torch.from_numpy(r_dic).float().to(self.device)

                self.model.update(act_p_dic, act_dic, r_dic, v_dic)
    
                if(e % 100 == 0 and e != 0):
                    self.model.save(e) 
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
        self.diff_state = torch.from_numpy(self.diff_state).float().to(self.device)
        action, _ = self.model.get_action(self.diff_state)
        state = prepro(observation)
        self.diff_state = state - self.pre_state
        self.pre_state = state
        return action

