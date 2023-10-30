from collections.abc import Callable, Iterable, Mapping
from typing import Any
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import torch.multiprocessing as mp
import copy 

from collections import namedtuple, deque

import gym
from tqdm import tqdm
from CartPole_modified import CartPole_fake
# from environments import SimpleDrone
# import actor-critics
from actor_critics import ActorCriticSNN_LIF_Small, ActorCritic_ANN,ActorCriticSNN_LIF_Smallest, ActorCritic_ANN_Smallest, ActorCritic_ANN_Cont, ActorCriticSNN_LIF_Smallest_Cont

def add_noise(state, gain=0.1):
    noise = np.random.normal(0, gain, state.shape)
    return state + noise
# env = gym.make("LunarLander-v2", render_mode="human")
# env.action_space.seed(42)

# observation, info = env.reset(seed=42)
# global model parameters
GAMMA = .98
LAMBDA_G = .9
ENTROPY_COEF = 1e-2
VALUE_LOSS_COEF = .5
POLICY_LOSS_COEF = 1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = None
MAX_GRAD_NORM = 1.5
WARM_UP = 0

EVALUATION_INTERVAL = 1000

'''
class MasterModel(mp.Process):
    def __init__(self, **args) -> None:
        super(MasterModel, self).__init__()
        self.game_name = 'CartPole'
        self.lr = args['lr']
        self.betas = args['betas']
        save_dir = args['save_dir']
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.env = CartPole_fake()
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space

        self.max_episodes = 10e3
        self.spiking = args['spiking']
        self.device  = args['device']
        self.args = args

        self.episode_times = []
        self.global_episode = 0  # Initialize the global episode counter

        self.noise_gain =  args['gain']
        if self.spiking:
            self.global_model = ActorCriticSNN_LIF_Small(self.state_size, self.action_size,
                                                   inp_min = torch.tensor([-4.8, -10,-0.418,-2]), 
                                                   inp_max=  torch.tensor([4.8, 10,0.418,2]), 
                                                   bias=False,nr_passes = 1).to(self.device)  # global network
        else:
            self.global_model = ActorCritic_ANN(self.state_size, self.action_size).to(self.device)  # global network
        # shared memory for gradients:
        # self.gradient_acc= torch.zeros(self.global_model.parameters()).share_memory()
        self.opt = torch.optim.Adam(self.global_model.parameters(), lr = args['lr'], betas= args['betas'])
        # nr_threads = 6
        

    def interact(self):
        env = self.env
        if self.spiking:
            self.global_model.init_mem() 
        

        # get initial state
        state, _ = env.reset()
        # state = env.set_state(constant_state)

        t_sim = 0
        terminal = False

        # print('Interacting with environment')
        while not terminal and t_sim < 500:
            state = add_noise(state, gain=self.noise_gain)
            # state to tensor
            state = torch.from_numpy(state).to(self.device)

            # get network outputs on given state
            value, policy = self.global_model(state.unsqueeze(0))

            # find probabilities of certain actions
            prob = F.softmax(policy, dim=-1)

            logprob = F.log_softmax(policy, dim=-1)

            # calculate entropy
            entropy = -(logprob * prob).sum(1, keepdim=True)

            # choose the action and detach from computational graph
            action = prob.multinomial(num_samples=1).detach() # find max of this and make sure it is not part of optimization

            # get log probs of the actions taken
            
            # perform action
            obs, reward, terminal, _, _ = env.step(int(action.squeeze(0)))


            if not terminal:
                state = obs

            t_sim += 1

        time_finish = t_sim   
    
        return time_finish, reward
    
    def run(self):
        EVALUATION_INTERVAL = 0
        nr_threads = mp.cpu_count()
        # mp.set_start_method('spawn')
        self.workers = [Worker(global_model=self.global_model, global_counter=self.global_episode, game_name=self.game_name,
                        idx=i, nr_workers=nr_threads, **self.args) for i in range(nr_threads)]
        for worker in self.workers:
            worker.start()

        with tqdm(total=self.max_episodes) as progress_bar:
            while self.global_episode <=self.max_episodes:
                if self.global_episode>EVALUATION_INTERVAL:
                    EVALUATION_INTERVAL += 100
                    # self.save_model()
                    t = 0
                    r = 0
                    for i in range(5):
                        t_cur,r_cur = self.interact()
                        t += t_cur
                        r += r_cur
                    t = t/5
                    r = r/5
                    self.episode_times.append(t)

                    print('Time to failure: ', t)
                    print('Reward: ', r)

                
                progress_bar.update(1)
                gradients = []
                for worker in self.workers:
                    # print('Starting next worker...')
                    gradients.append(worker.run())

                    # gradients.append(worker.run())
                # now synchronous
                results = []
                for i,tensors in enumerate(zip(*gradients)):
                    if tensors[0] is not None:
                        results.append(sum(tensors)/nr_threads)
                    else:
                        results.append(None)

                # result_list = [sum(tensors)/nr_threads for tensors in zip(*gradients)]
   
                self.opt.zero_grad()
                for i,param in enumerate(self.global_model.parameters()):
                    param.grad = results[i] 
                self.opt.step()
                # print(len(gradients[0]), len(gradients[1]))
                for worker in self.workers:
                    # print('\nJoining threads...')
                    worker.join()
                    

                self.global_episode += 1
                
    def save_model(self):
        torch.save(self.global_model.state_dict(), self.save_dir + '/model_'+self.game_name+'.pt')
        print("Saved Model")


class Worker(mp.Process):
    global_episode = 0
    gloabl_MA      = 0
    best_score     = 0
    save_lock      = mp.Lock()
    t_sim_max      = 2000
    total_runs     = int(10e3)

    def __init__(self, global_model, global_counter, game_name, save_dir, nr_workers, **args):
        super(Worker, self).__init__(group=None)
        self.global_model = global_model
        
        self.global_episode = global_counter
        self.game_name = game_name
        self.save_dir = save_dir
        self.ep_loss = 0.0

        self.env = CartPole_fake()
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space

        self.noise_gain = args['gain']
        self.spiking     = args['spiking']
        self.device      = args['device']
        print('NOT CLAMPING REWARD')
        print("ACTION IS POLICY")
        # for loss aggregation
        self.nr_workers = nr_workers
        if self.spiking:
            self.local_model = ActorCriticSNN_LIF_Small(self.state_size, self.action_size,
                                                   inp_min = torch.tensor([-4.8, -10,-0.418,-2]), 
                                                   inp_max=  torch.tensor([4.8, 10,0.418,2]), 
                                                   bias=False,nr_passes = 1).to(self.device)  # global network
            # self.local_model = ActorCriticSNN_LIF_Small(self.state_size, self.action_size,
            #                                        inp_min = torch.tensor([-4.8, -10,-0.418,-2]), 
            #                                        inp_max=  torch.tensor([4.8, 10,0.418,2]), 
            #                                        bias=False,nr_passes = 1).to(self.device)  # global network
            self.local_model.load_state_dict(self.global_model.state_dict())

        else:
            self.local_model = ActorCritic_ANN(self.state_size, self.action_size).to(self.device)  # global network
            self.local_model.load_state_dict(self.global_model.state_dict())

        self.opt = self.opt = torch.optim.Adam(self.local_model.parameters(), 
                                               lr = args['lr'], betas= args['betas'])
        # for plotting purposes:
        self.T_lst = [] # list of times in environment

        self.train_cycles = 0

    def interact(self):
        env = self.env

        self.local_model.load_state_dict(self.global_model.state_dict())

        if self.spiking:
            self.local_model.init_mem() 
        

        # get initial state
        state, _ = env.reset()
        # state = env.set_state(constant_state)

        t_sim = 0
        terminal = False
        reward = 0

        values = []
        log_probs = []
        rewards = []
        entropies = []
        # print('Interacting with environment')
        while not terminal and t_sim < Worker.t_sim_max:
            state = add_noise(state, gain=self.noise_gain)
            # state to tensor
            state = torch.from_numpy(state).to(self.device)

            # get network outputs on given state
            value, policy = self.local_model(state.unsqueeze(0))

            # find probabilities of certain actions
            prob = F.softmax(policy, dim=-1)

            logprob = F.log_softmax(policy, dim=-1)

            # calculate entropy
            entropy = -(logprob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            # choose the action and detach from computational graph
            action = prob.multinomial(num_samples=1).detach() # find max of this and make sure it is not part of optimization
            # get log probs of the actions taken
            log_prob = logprob.gather(1,action)

            # perform action
            obs, reward, terminal, _, _ = env.step(int(action.squeeze(0)))

            # !!!!!!!!!!!
   
            # reward = max(min(reward, 1), -1)

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            
            if not terminal:
                state = obs

            t_sim += 1

        time_finish = t_sim   
    
        R = 0
        if not terminal:
            R, _ = self.local_model(state.to(self.device))
            R = R.detach()

            
        # save current R (value) for the gneralized advantage DeltaT
        prev_val = R
        values.append(R)
        self.T_lst.append(time_finish)

        return {'values':values, 'rewards':rewards, 'log_probs':log_probs, 'entropies':entropies, 'prev_val': R}
    
    def calculate_loss(self, values, rewards, log_probs, entropies, last_val):
        R = last_val

        policy_loss = 0
        value_loss  = 0
        g = torch.zeros(1,1).to(self.device)
        len_counter = 0
        # note cycle through memory sequentially as we are appending left
        for i in reversed(range(len(rewards))):
            len_counter+=1

            if len_counter>WARM_UP:

                R = rewards[i] + GAMMA*R

                advantage = R - values[i] # note that according to the git they are out of sync by one step 

                # alternative training loop
                value_loss = value_loss + 0.5*advantage.pow(2)

                # generalized advantage estimation from: https://arxiv.org/pdf/1506.02438.pdf
                delta_t = rewards[i] + GAMMA* values[i+1] - values[i]
                g = g*GAMMA*LAMBDA_G + delta_t

                policy_loss = policy_loss - log_probs[i]*g.detach() - entropies[i]*ENTROPY_COEF

            
            else:
                value_loss = 0
                policy_loss = 0
        if self.spiking:
            spike_sparsity_loss = torch.sum(torch.stack(self.local_model.spk_in_rec)) / len_counter
            spike_sparsity_loss = 0
        else:
            spike_sparsity_loss = 0

        loss_eval = (policy_loss * POLICY_LOSS_COEF+ value_loss * VALUE_LOSS_COEF ) / len_counter
        loss_eval += (spike_sparsity_loss*1e-3 + 1/(spike_sparsity_loss+1))

        return loss_eval
        
    def train(self):
        memory = self.interact()

        values = memory['values']
        rewards = memory['rewards']
        log_probs = memory['log_probs']
        entropies = memory['entropies']
        last_val  = memory['prev_val']
        loss = self.calculate_loss(values,rewards, log_probs, entropies, last_val)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), MAX_GRAD_NORM)
        

        # find a way to accumulate all gradients
        gradients = [param.grad for param in self.local_model.parameters()]
        return gradients



    def run(self):
        with self.save_lock:
            self.global_episode += 1
        self.train_cycles += 1
        gradients = self.train()
            # barrier.wait()  # Wait for all workers to finish this iteration

        # def moving_average(a, n=50):
        #     ret = np.cumsum(a, dtype=float)
        #     ret[n:] = ret[n:] - ret[:-n]
        #     return ret[n - 1:] / n
        
        # self.T_lst = moving_average(self.T_lst)
        return gradients

        # plt.plot(range(len(self.T_lst)),self.T_lst)
        # plt.show()
'''
class MasterModel(mp.Process):
    def __init__(self, **args) -> None:
        super(MasterModel, self).__init__()
        self.game_name = 'CartPole-v1'
        self.lr = args['lr']
        self.betas = args['betas']
        save_dir = args['save_dir']
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.dt = args['dt']
        self.env = CartPole_fake(dt=self.dt)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space
        
        self.max_episodes = args['nr_episodes']
        self.max_length = args['max_episode_length']
        self.spiking = args['spiking']
        self.device  = args['device']
        self.args = args
        self.norm = args['normalize_steps']
        self.episode_times = []
        self.global_episode = 0  # Initialize the global episode counter

        if self.spiking:
            self.global_model = ActorCriticSNN_LIF_Smallest(self.state_size, self.action_size,
                                                   inp_min = torch.tensor([-4.8, -10,-0.418,-2]), 
                                                   inp_max=  torch.tensor([4.8, 10,0.418,2]), 
                                                   bias=False,nr_passes = 1).to(self.device)  # global network
        else:
            self.global_model = ActorCritic_ANN_Smallest(self.state_size, self.action_size).to(self.device)  # global network
        # shared memory for gradients:
        # self.gradient_acc= torch.zeros(self.global_model.parameters()).share_memory()
        self.opt = torch.optim.Adam(self.global_model.parameters(), lr = args['lr'], betas= args['betas'])
        # nr_threads = 6
        

    def interact(self):
        env = self.env
        if self.spiking:
            self.global_model.init_mem() 
        

        # get initial state
        state, _ = env.reset()
        # state = env.set_state(constant_state)

        t_sim = 0
        terminal = False

        # print('Interacting with environment')
        while not terminal and t_sim < self.max_length:

            # state to tensor
            state = torch.from_numpy(state).to(self.device)

            # get network outputs on given state
            value, policy = self.global_model(state.unsqueeze(0))

            # find probabilities of certain actions
            prob = F.softmax(policy, dim=-1)

            logprob = F.log_softmax(policy, dim=-1)

            # calculate entropy
            entropy = -(logprob * prob).sum(1, keepdim=True)

            # choose the action and detach from computational graph
            action = prob.multinomial(num_samples=1).detach() # find max of this and make sure it is not part of optimization

            # get log probs of the actions taken
            
            # perform action
            obs, reward, terminal, _, _ = env.step(int(action.squeeze(0)))


            if not terminal:
                state = obs

            t_sim += 1

        time_finish = t_sim   
    
        return time_finish
    
    def run(self):
        EVALUATION_INTERVAL = 0
        nr_threads = mp.cpu_count()
        # mp.set_start_method('spawn')
        self.workers = [Worker(global_model=self.global_model, global_counter=self.global_episode, game_name=self.game_name,
                        idx=i, nr_workers=nr_threads, **self.args) for i in range(nr_threads)]
        for worker in self.workers:
            worker.start()

        with tqdm(total=self.max_episodes) as progress_bar:
            while self.global_episode <=self.max_episodes:
                if self.global_episode>EVALUATION_INTERVAL:
                    EVALUATION_INTERVAL += 100
                    # self.save_model()
                    t = 0
                    for i in range(5):
                        t += self.interact()
                    t = t/5
                    self.episode_times.append(t)

                    print('Time to failure: ', t)

                
                progress_bar.update(1)
                gradients = []
                total_timesteps = 0
                # we want to normalize timesteps, reduce impact of shittier series with short time and increase with more time
                # unbiased way is t1*grad1 + t2*grad2 + ... / t1 + t2 + ...
                for worker in self.workers:
                    # print('Starting next worker...')
                    gradient, timesteps = worker.run()
                    gradients.append(gradient* timesteps)
                    total_timesteps += timesteps

                    # gradients.append(worker.run())
                # now synchronous
                results = []
                for i,tensors in enumerate(zip(*gradients)):
                    if tensors[0] is not None:
                        if self.norm:
                            results.append(sum(tensors)/total_timesteps)
                        else:
                            results.append(sum(tensors)/nr_threads)

                    else:
                        results.append(None)

                # result_list = [sum(tensors)/nr_threads for tensors in zip(*gradients)]
   
                self.opt.zero_grad()
                for i,param in enumerate(self.global_model.parameters()):
                    param.grad = results[i] 
                self.opt.step()
                # print(len(gradients[0]), len(gradients[1]))
                for worker in self.workers:
                    # print('\nJoining threads...')
                    worker.join()
                    

                self.global_episode += 1
                
    def save_model(self, path=None):
        if path is None:
            torch.save(self.global_model.state_dict(), self.save_dir + '/model_'+self.game_name+'.pt')
        else:
            torch.save(self.global_model.state_dict(),path)
        print("Saved Model")


class Worker(mp.Process):
    # global_episode = 0
    gloabl_MA      = 0
    best_score     = 0
    save_lock      = mp.Lock()
    # total_runs     = int(25e3)

    def __init__(self, global_model, global_counter, game_name, save_dir, nr_workers, **args):
        super(Worker, self).__init__(group=None)
        self.global_model = global_model

        self.global_episode = global_counter
        self.game_name = game_name
        self.dt = args['dt']
        self.env = CartPole_fake(self.dt)
        self.save_dir = save_dir
        self.t_sim_max = args['max_episode_length']
        self.total_runs = args['nr_episodes']
        self.ep_loss = 0.0
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space
        self.spiking     = args['spiking']
        self.device      = args['device']
        self.noise_gain  = args['gain']

        # for loss aggregation
        self.nr_workers = nr_workers
        if self.spiking:
            self.local_model = ActorCriticSNN_LIF_Smallest(self.state_size, self.action_size,
                                                   inp_min = torch.tensor([-4.8, -10,-0.418,-2]), 
                                                   inp_max=  torch.tensor([4.8, 10,0.418,2]), 
                                                   bias=False,nr_passes = 1).to(self.device)  # global network
            self.local_model.load_state_dict(self.global_model.state_dict())

        else:
            self.local_model = ActorCritic_ANN_Smallest(self.state_size, self.action_size).to(self.device)  # global network
            self.local_model.load_state_dict(self.global_model.state_dict())

        self.opt = self.opt = torch.optim.Adam(self.local_model.parameters(), 
                                               lr = args['lr'], betas= args['betas'])
        # for plotting purposes:
        self.T_lst = [] # list of times in environment

        self.train_cycles = 0

    def interact(self):
        env = self.env

        self.local_model.load_state_dict(self.global_model.state_dict())

        if self.spiking:
            self.local_model.init_mem() 
        

        # get initial state
        state, _ = env.reset()
        # state = env.set_state(constant_state)

        t_sim = 0
        terminal = False
        reward = 0

        values = []
        log_probs = []
        rewards = []
        entropies = []
        # print('Interacting with environment')
        while not terminal and t_sim < self.t_sim_max:
            state = add_noise(state, gain=self.noise_gain)
            # state to tensor
            state = torch.from_numpy(state).to(self.device)

            # get network outputs on given state
            value, policy = self.local_model(state.unsqueeze(0))

            # find probabilities of certain actions
            prob = F.softmax(policy, dim=-1)

            logprob = F.log_softmax(policy, dim=-1)

            # calculate entropy
            entropy = -(logprob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            # choose the action and detach from computational graph
            action = prob.multinomial(num_samples=1).detach() # find max of this and make sure it is not part of optimization

            # get log probs of the actions taken
            log_prob = logprob.gather(1,action)
            
            # perform action
            obs, reward, terminal, _, _ = env.step(int(action.squeeze(0)))

            # !!!!!!!!!!!
            reward = max(min(reward, 1), -1)

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            
            if not terminal:
                state = obs

            t_sim += 1

        time_finish = t_sim   
    
        R = 0
        if not terminal:
            R, _ = self.local_model(torch.from_numpy(state).to(self.device))
            R = R.detach()

            
        # save current R (value) for the gneralized advantage DeltaT
        prev_val = R
        values.append(R)
        self.T_lst.append(time_finish)

        return {'values':values, 'rewards':rewards, 'log_probs':log_probs, 'entropies':entropies, 'prev_val': R, 'timesteps': time_finish}
    
    def calculate_loss(self, values, rewards, log_probs, entropies, last_val):
        R = last_val

        policy_loss = 0
        value_loss  = 0
        g = torch.zeros(1,1).to(self.device)
        len_counter = 0
        # note cycle through memory sequentially as we are appending left
        for i in reversed(range(len(rewards))):
            len_counter+=1

            if len_counter>WARM_UP:

                R = rewards[i] + GAMMA*R

                advantage = R - values[i] # note that according to the git they are out of sync by one step 

                # alternative training loop
                value_loss = value_loss + 0.5*advantage.pow(2)

                # generalized advantage estimation from: https://arxiv.org/pdf/1506.02438.pdf
                delta_t = rewards[i] + GAMMA* values[i+1] - values[i]
                g = g*GAMMA*LAMBDA_G + delta_t

                policy_loss = policy_loss - log_probs[i]*g.detach() - entropies[i]*ENTROPY_COEF

            
            else:
                value_loss = 0
                policy_loss = 0

        # spike_sparsity_loss = torch.sum(torch.stack(self.local_model.spk_in_rec)) / len_counter

        loss_eval = (policy_loss * POLICY_LOSS_COEF+ value_loss * VALUE_LOSS_COEF ) / len_counter
        # loss_eval += (spike_sparsity_loss*1e-3 + 1/(spike_sparsity_loss+1))

        return loss_eval
        
    def train(self):
        memory = self.interact()

        values = memory['values']
        rewards = memory['rewards']
        log_probs = memory['log_probs']
        entropies = memory['entropies']
        last_val  = memory['prev_val']
        loss = self.calculate_loss(values,rewards, log_probs, entropies, last_val)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), MAX_GRAD_NORM)
        

        # find a way to accumulate all gradients
        gradients = [param.grad for param in self.local_model.parameters()]
        return gradients, memory['timesteps']



    def run(self):
        with self.save_lock:
            self.global_episode += 1
        self.train_cycles += 1
        gradients, timesteps = self.train()
            # barrier.wait()  # Wait for all workers to finish this iteration

        # def moving_average(a, n=50):
        #     ret = np.cumsum(a, dtype=float)
        #     ret[n:] = ret[n:] - ret[:-n]
        #     return ret[n - 1:] / n
        
        # self.T_lst = moving_average(self.T_lst)
        return gradients,timesteps



class MasterModel_continuous(mp.Process):
    def __init__(self, **args) -> None:
        super(MasterModel_continuous, self).__init__()
        self.game_name = 'Pendulum-v1'
        self.lr = args['lr']
        self.betas = args['betas']
        save_dir = args['save_dir']
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.env = gym.make(self.game_name)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        # self.env = SimpleDrone()
        # self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]


        self.max_episodes = args['nr_episodes']
        self.spiking = args['spiking']
        self.device  = args['device']
        self.args = args
        self.std = np.log(.2)
        self.episode_times = []
        self.global_episode = 0  # Initialize the global episode counter

        if self.spiking:
            self.global_model = ActorCriticSNN_LIF_Smallest_Cont(self.state_size, self.action_size,
                                                #    inp_min = torch.tensor([-4.8, -10,-0.418,-2]), 
                                                #    inp_max=  torch.tensor([4.8, 10,0.418,2]), 
                                                   inp_min= torch.tensor([0]),
                                                   inp_max =torch.tensor([2]),
                                                   bias=False,nr_passes = 1).to(self.device)  # global network
        else:
            self.global_model = ActorCritic_ANN_Cont(self.state_size, self.action_size).to(self.device)  # global network
        # shared memory for gradients:
        # self.gradient_acc= torch.zeros(self.global_model.parameters()).share_memory()
        self.opt = torch.optim.Adam(self.global_model.parameters(), lr = args['lr'], betas= args['betas'])
        # nr_threads = 6
        

    def interact(self):
        env = self.env
        if self.spiking:
            self.global_model.init_mem() 
        

        # get initial state
        state, _ = env.reset()
        # state = env.set_state(constant_state)

        t_sim = 0
        terminal = False

        # print('Interacting with environment')
        while not terminal and t_sim < 500:

            # state to tensor
            state = torch.from_numpy(state).to(self.device)

            # get network outputs on given state
            _, mean, _ = self.global_model(state.unsqueeze(0))
            log_std = torch.ones_like(mean)*self.std
            std = torch.max(torch.exp(log_std), torch.tensor(0.01))
            # Sample an action from a Gaussian distribution defined by mean and standard deviation
            action = torch.normal(mean, std).detach()
            # print(action)
            # perform action
            obs, reward, terminal, _, _ = env.step(action.squeeze(0))


            if not terminal:
                state = obs

            t_sim += 1

        time_finish = t_sim   
    
        return time_finish, reward
    
    def run(self):
        EVALUATION_INTERVAL = 0
        nr_threads = mp.cpu_count()
        # mp.set_start_method('spawn')
        self.workers = [Worker_continuous(global_model=self.global_model, global_counter=self.global_episode, game_name=self.game_name,
                        idx=i, nr_workers=nr_threads, **self.args) for i in range(nr_threads)]
        for worker in self.workers:
            worker.start()

        with tqdm(total=self.max_episodes) as progress_bar:
            while self.global_episode <=self.max_episodes:
                if self.global_episode>EVALUATION_INTERVAL:
                    EVALUATION_INTERVAL += 100
                    # self.save_model()
                    t = 0
                    r = 0
                    print('Evaluation')
                    for i in range(5):
                        t_cur,r_cur = self.interact()
                        t += t_cur
                        r += r_cur
                    t = t/5
                    r = r/5
                    self.episode_times.append(t)

                    print('Time to failure: ', t)
                    print('Reward: ', r)
                    # if t>200:
                    #     self.env.plot()

                
                progress_bar.update(1)
                gradients = []
                for worker in self.workers:
                    # print('Starting next worker...')
                    gradients.append(worker.run())

                    # gradients.append(worker.run())
                # now synchronous
                results = []
                for i,tensors in enumerate(zip(*gradients)):
                    if tensors[0] is not None:
                        results.append(sum(tensors)/nr_threads)
                    else:
                        results.append(None)

                # result_list = [sum(tensors)/nr_threads for tensors in zip(*gradients)]
   
                self.opt.zero_grad()
                for i,param in enumerate(self.global_model.parameters()):
                    param.grad = results[i] 
                self.opt.step()
                # print(len(gradients[0]), len(gradients[1]))
                for worker in self.workers:
                    # print('\nJoining threads...')
                    worker.join()
                    

                self.global_episode += 1
                
    def save_model(self):
        torch.save(self.global_model.state_dict(), self.save_dir + '/model_'+self.game_name+'.pt')
        print("Saved Model")


class Worker_continuous(mp.Process):
    # global_episode = 0
    gloabl_MA      = 0
    best_score     = 0
    save_lock      = mp.Lock()
    

    def __init__(self, global_model, global_counter, game_name, save_dir, nr_workers, **args):
        super(Worker_continuous, self).__init__(group=None)
        self.global_model = global_model

        self.global_episode = global_counter
        self.game_name = game_name
        self.save_dir = save_dir
        self.ep_loss = 0.0

        self.total_runs = args['nr_episodes']
        self.t_sim_max = args['max_episode_length']

        self.env = gym.make(self.game_name)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]

        self.log_std = np.log(0.2)
        # self.env = SimpleDrone()
        # self.state_size = self.env.observation_space.shape[0]
        # self.action_size = self.env.action_space.shape[0]

        self.spiking     = args['spiking']
        self.device      = args['device']
        print('NOT CLAMPING REWARD')
        # for loss aggregation
        self.nr_workers = nr_workers
        if self.spiking:
            self.local_model = ActorCriticSNN_LIF_Smallest_Cont(self.state_size, self.action_size,
                                    #    inp_min = torch.tensor([-4.8, -10,-0.418,-2]), 
                                    #    inp_max=  torch.tensor([4.8, 10,0.418,2]), 
                                                   inp_min= torch.tensor([0,-.5]),
                                                   inp_max =torch.tensor([2,.5]),
                                                   bias=False,nr_passes = 1).to(self.device)  # global network
            # self.local_model = ActorCriticSNN_LIF_Small(self.state_size, self.action_size,
            #                                        inp_min = torch.tensor([-4.8, -10,-0.418,-2]), 
            #                                        inp_max=  torch.tensor([4.8, 10,0.418,2]), 
            #                                        bias=False,nr_passes = 1).to(self.device)  # global network
            self.local_model.load_state_dict(self.global_model.state_dict())

        else:
            self.local_model = ActorCritic_ANN_Cont(self.state_size, self.action_size).to(self.device)  # global network
            self.local_model.load_state_dict(self.global_model.state_dict())

        self.opt = self.opt = torch.optim.Adam(self.local_model.parameters(), 
                                               lr = args['lr'], betas= args['betas'])
        # for plotting purposes:
        self.T_lst = [] # list of times in environment

        self.train_cycles = 0

    def interact(self):
        env = self.env

        self.local_model.load_state_dict(self.global_model.state_dict())

        if self.spiking:
            self.local_model.init_mem() 
        

        # get initial state
        state, _ = env.reset()
        # state = env.set_state(constant_state)

        t_sim = 0
        terminal = False
        reward = 0

        values = []
        log_probs = []
        rewards = []
        entropies = []
        # print('Interacting with environment')
        while not terminal and t_sim < self.t_sim_max:

            # state to tensor
            state = torch.tensor(state).to(self.device)

            # get network outputs on given state
            value, mean, _ = self.local_model(state.unsqueeze(0))
            log_std = self.log_std*torch.ones_like(mean)
            std = torch.clamp(torch.exp(log_std), torch.tensor(0.01),torch.tensor(25))
            mean = torch.clamp(mean, torch.tensor(-25),torch.tensor(25))
            # Sample an action from a Gaussian distribution defined by mean and standard deviation
            # print(std)

            action = torch.normal(mean, std=std)
         
                
            print(-0.5 * (log_std + (action - mean).pow(2) / torch.exp(log_std)))
            print((-0.5 * (log_std + (action - mean).pow(2) / torch.exp(log_std))).shape)
            # Calculate the log probability of the selected action
            log_prob = -0.5 * (log_std + (action - mean).pow(2) / torch.exp(log_std)).sum(1, keepdim=True)

            # entropy = 0.5 * (log_std + 1.0 + 0.5 * torch.log(torch.tensor(2 * torch.pi))).sum(1, keepdim=True)

            entropy = -0.5 * (log_std + 1.0 + 0.5 * torch.log(torch.tensor(2 * torch.pi))).sum(1, keepdim=True)
            entropies.append(entropy)

       
            # perform action
            obs, reward, terminal, _, _ = env.step(action.squeeze(0).detach().numpy())
            # !!!!!!!!!!!
   
            reward = max(min(reward, 1), -1)

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            
            if not terminal:
                state = obs

            t_sim += 1

        time_finish = t_sim   
    
        R = 0
        if not terminal:
            state = torch.tensor(state).to(self.device)
            R, _,_ = self.local_model(state)
            R = R.detach()

            
        # save current R (value) for the gneralized advantage DeltaT
        prev_val = R
        values.append(R)
        self.T_lst.append(time_finish)

        return {'values':values, 'rewards':rewards, 'log_probs':log_probs, 'entropies':entropies, 'prev_val': R}
    
    def calculate_loss(self, values, rewards, log_probs, entropies, last_val):
        R = last_val

        policy_loss = 0
        value_loss  = 0
        g = torch.zeros(1,1).to(self.device)
        len_counter = 0
        # note cycle through memory sequentially as we are appending left
        for i in reversed(range(len(rewards))):
            len_counter+=1

            if len_counter>WARM_UP:

                R = rewards[i] + GAMMA*R

                advantage = R - values[i] # note that according to the git they are out of sync by one step 

                # alternative training loop
                value_loss = value_loss + 0.5*advantage.pow(2)

                # generalized advantage estimation from: https://arxiv.org/pdf/1506.02438.pdf
                delta_t = rewards[i] + GAMMA* values[i+1] - values[i]
                g = g*GAMMA*LAMBDA_G + delta_t

                policy_loss = policy_loss - log_probs[i]*g.detach() - entropies[i]*ENTROPY_COEF

            
            else:
                value_loss = 0
                policy_loss = 0
        if self.spiking:
            spike_sparsity_loss = torch.sum(torch.stack(self.local_model.spk_in_rec)) / len_counter
            spike_sparsity_loss = 0
        else:
            spike_sparsity_loss = 0

        loss_eval = (policy_loss * POLICY_LOSS_COEF+ value_loss * VALUE_LOSS_COEF ) / len_counter
        # loss_eval += (spike_sparsity_loss*1e-3 + 1/(spike_sparsity_loss+1))

        return loss_eval
        
    def train(self):
        memory = self.interact()

        values = memory['values']
        rewards = memory['rewards']
        log_probs = memory['log_probs']
        entropies = memory['entropies']
        last_val  = memory['prev_val']
        loss = self.calculate_loss(values,rewards, log_probs, entropies, last_val)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), MAX_GRAD_NORM)
        

        # find a way to accumulate all gradients
        gradients = [param.grad for param in self.local_model.parameters()]
        return gradients



    def run(self):
        with self.save_lock:
            self.global_episode += 1
        self.train_cycles += 1
        gradients = self.train()
            # barrier.wait()  # Wait for all workers to finish this iteration

        # def moving_average(a, n=50):
        #     ret = np.cumsum(a, dtype=float)
        #     ret[n:] = ret[n:] - ret[:-n]
        #     return ret[n - 1:] / n
        
        # self.T_lst = moving_average(self.T_lst)
        return gradients

        # plt.plot(range(len(self.T_lst)),self.T_lst)
        # plt.show()
# 