from collections.abc import Callable, Iterable, Mapping
from typing import Any
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import torch.multiprocessing as mp
import torch.multiprocessing.queue as mpq
import copy 

from collections import namedtuple, deque

import gym
from tqdm import tqdm
# from CartPole_modified import CartPole_fake
from environments import SimpleDrone_Discrete
# import actor-critics
from actor_critics import ActorCriticSNN_LIF_Small, ActorCritic_ANN,ActorCriticSNN_LIF_Smallest, ActorCritic_ANN_Smallest, ActorCritic_ANN_Cont, ActorCriticSNN_LIF_Smallest_Cont,ActorCriticSNN_LIF_drone

def add_noise(state, gain=0.1):
    noise = np.random.normal(0, gain, state.shape)
    return state + noise
# env = gym.make("LunarLander-v2", render_mode="human")
# env.action_space.seed(42)

# observation, info = env.reset(seed=42)
# global model parameters
GAMMA = .98
LAMBDA_G = .9
ENTROPY_COEF = 1
ENTROPY_COEF_end = 1e-3
VALUE_LOSS_COEF = .5
POLICY_LOSS_COEF = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = None
MAX_GRAD_NORM = 1.5
WARM_UP = 0

EVALUATION_INTERVAL = 1000

class MasterModel(mp.Process):
    def __init__(self, **args) -> None:
        super(MasterModel, self).__init__()
        self.game_name = 'Drone'
        self.lr = args['lr']
        self.betas = args['betas']
        save_dir = args['save_dir']
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.dt = args['dt']
        self.max_length = args['max_episode_length']

        # self.env = CartPole_fake(dt=self.dt)
        self.env = SimpleDrone_Discrete(dt=self.dt, max_episode_length=self.max_length)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space
        
        self.max_episodes = args['nr_episodes']
        self.spiking = args['spiking']
        self.device  = args['device']
        self.args = args
        self.norm = args['normalize_steps']
        self.episode_times = []
        self.global_episode = 0  # Initialize the global episode counter

        if self.spiking:
            self.global_model = ActorCriticSNN_LIF_drone(self.state_size, self.action_size, hidden1=32, hidden2=32).to(self.device)  # global network

            # self.global_model = ActorCriticSNN_LIF_Smallest(self.state_size, self.action_size,
            #                                     #    inp_min = torch.tensor([-4.8, -10,-0.418,-2]), 
            #                                     #    inp_max=  torch.tensor([4.8, 10,0.418,2]), 
            #                                     #    inp_min = torch.tensor([0, -.5]), 
            #                                     #    inp_max=  torch.tensor([2.5, .5]), 
            #                                         inp_min = torch.tensor([0]), 
            #                                        inp_max=  torch.tensor([2.5]), 
            #                                        bias=False,nr_passes = 1).to(self.device)  # global network
        else:
            self.global_model = ActorCritic_ANN(self.state_size, self.action_size).to(self.device)  # global network
        # shared memory for gradients:
        # self.gradient_acc= torch.zeros(self.global_model.parameters()).share_memory()
        self.opt = torch.optim.Adam(self.global_model.parameters(), lr = args['lr'], betas= args['betas'])
        

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
            # state, vel = state[0], state[1] # new
# get network outputs on given state
            _, policy,_ = self.global_model(state.unsqueeze(0))

            # find probabilities of certain actions
            prob = F.softmax(policy, dim=-1)


            # choose the action and detach from computational graph
            action = prob.multinomial(num_samples=1).detach() # find max of this and make sure it is not part of optimization
            
            # perform action
            obs, reward, terminal, _, _ = env.step(int(action.squeeze(0)))


            if not terminal:
                state = obs

            
            t_sim += 1

        time_finish = t_sim   
    
        return time_finish, reward
    
    def run(self):
        EVALUATION_INTERVAL = 500
        nr_threads = mp.cpu_count()
        nr_threads = 4
        print('\nCreating workers...')
        self.workers = [Worker(global_model=self.global_model, global_counter=self.global_episode, game_name=self.game_name,
                        idx=i, nr_workers=nr_threads, **self.args) for i in range(nr_threads)]
        
        print('\nStarting workers...')
        for worker in self.workers:
            worker.start()

        total_eps = self.max_episodes
        progress_bar = tqdm(total=total_eps)

        for _ in range(int(total_eps)):
            progress_bar.update(1)
            self.global_episode += 1 
            Worker.global_MA += 1
            if self.global_episode>EVALUATION_INTERVAL:
                EVALUATION_INTERVAL += 5000
                # self.save_model(path='intermediate_savings/model_'+str(EVALUATION_INTERVAL)+'.pt')
                t = 0
                reward = 0
                for i in range(5):
                    t_cur, reward_cur = self.interact()
                    # landing += info['landing']*1
                    t += t_cur
                    reward += reward_cur
                t = t/5
                reward = reward/5

                self.episode_times.append(t)
                

                print('Time to failure: ', t)
                print('Reward: ', reward)

            
            
            gradients = []
            total_timesteps = 0
            # we want to normalize timesteps, reduce impact of shittier series with short time and increase with more time
            # unbiased way is t1*grad1 + t2*grad2 + ... / t1 + t2 + ...
            # with mp.Pool(processes=nr_threads) as pool:
            #     results = pool.map(Worker.run, self.workers)
            # gradients, total_timesteps = zip(*results)

            # # Process the results as needed
            # gradients = [grad for grad in gradients]
            # total_timesteps = sum(total_timesteps)
            for worker in self.workers:
                gradient, timesteps = worker.run()
                gradients.append([grad  for grad in gradient])
                total_timesteps += timesteps

            # now synchronous              
            results = []
            for i,tensors in enumerate(zip(*gradients)):
                if tensors[0] is not None:
                    if self.norm:
                        # Stack the tensors along a new dimension (default is dim=0)
                        # stacked_tensor = torch.stack(tensors)

                        # Sum along the newly added dimension to get the final tensor
                        results.append(sum(tensors)/total_timesteps)

                    else:
                        results.append(sum(tensors)/nr_threads)

                else:
                    results.append(None)


            self.opt.zero_grad()
            for i,param in enumerate(self.global_model.parameters()):
                param.grad = results[i] 
            self.opt.step()
            self.global_model.clip_hiddens() # assure everything beteen 0 and 1

            for worker in self.workers:
                worker.join()
                

            # update entropy loss
            # ENTROPY_COEF = max(1,min(ENTROPY_COEF_end,ENTROPY_COEF_end + (ENTROPY_COEF - ENTROPY_COEF_end)*np.exp(-1e-5*self.global_episode)))
                
    def save_model(self, path=None):
        if path is None:
            torch.save(self.global_model.state_dict(), self.save_dir + '/model_'+self.game_name+'.pt')
        else:
            torch.save(self.global_model.state_dict(),path)
        print("Saved Model")


class Worker(mp.Process):
    instance_count = 0
    best_score     = 0
    save_lock      = mp.Lock()
    global_MA = 0
    # total_runs     = int(25e3)

    def __init__(self, global_model, global_counter, game_name, save_dir, nr_workers, **args):
        super(Worker, self).__init__(group=None)
        Worker.instance_count += 1
        print('Creating worker nr: ', Worker.instance_count)
        self.global_model = global_model

        self.game_name = game_name
        self.dt = args['dt']
        self.t_sim_max = args['max_episode_length']

        # self.env = CartPole_fake(self.dt)
        self.env = SimpleDrone_Discrete(dt=self.dt, max_episode_length=self.t_sim_max)
        self.save_dir = save_dir
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
            self.local_model = ActorCriticSNN_LIF_drone(self.state_size, self.action_size, hidden1=32, hidden2=32).to(self.device)  # global network
            # self.local_model = ActorCriticSNN_LIF_Smallest(self.state_size, self.action_size,
            #                                     #    inp_min = torch.tensor([-4.8, -10,-0.418,-2]), 
            #                                     #    inp_max=  torch.tensor([4.8, 10,0.418,2]), 
            #                                        inp_min = torch.tensor([0]), 
            #                                        inp_max=  torch.tensor([2.5]), 
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
        print('Now nr episodes is nr of updates of global model')


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
        velocities = [] # new
        predicted_velocities = [] # new
        # print('Interacting with environment')
        while not terminal and t_sim < self.t_sim_max:
            state = add_noise(state, gain=self.noise_gain)
            # state to tensor
            state = torch.from_numpy(state).to(self.device)
            # state, velocity = state[0], state[1] # new
            # get network outputs on given state
            value, policy, vel = self.local_model(state.unsqueeze(0)) # new
            # predicted_velocities.append(vel) # new
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
            # velocities.append(velocity) # new

            
            if not terminal:
                state = obs

            t_sim += 1

        time_finish = t_sim   
    
        R = 0
        if not terminal:
            R, _,_ = self.local_model(torch.from_numpy(state).to(self.device))
            R = R.detach()

            
        # save current R (value) for the gneralized advantage DeltaT
        prev_val = R
        values.append(R)
        self.T_lst.append(time_finish)

        return {'values':values, 'rewards':rewards, 'log_probs':log_probs, 'entropies':entropies, 'prev_val': R, 'timesteps': time_finish, 'velocities': velocities, 'predicted_velocities': predicted_velocities}
    
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

                curr_entropy_coeff = max(1,min(ENTROPY_COEF_end,ENTROPY_COEF_end + (ENTROPY_COEF - ENTROPY_COEF_end)*np.exp(-1e-5*Worker.global_MA)))
                policy_loss = policy_loss - log_probs[i]*g.detach() - entropies[i]*curr_entropy_coeff

            
            else:
                value_loss = 0
                policy_loss = 0

        # spike_sparsity_loss = torch.sum(torch.stack(self.local_model.spk_in_rec)) / len_counter

        loss_eval = (policy_loss * POLICY_LOSS_COEF+ value_loss * VALUE_LOSS_COEF ) / len_counter
        # loss_eval += (spike_sparsity_loss*1e-3 + 1/(spike_sparsity_loss+1))

        return loss_eval
        
    def calculate_vel_loss(self, velocities, predicted_velocities):
        loss_function = torch.nn.MSELoss()
        predicted_velocities = predicted_velocities.reshape(-1)

        loss_val = loss_function(predicted_velocities.float(), velocities.float()) # calculate loss
        
        return loss_val
    

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

    def train_velocity(self):
        memory = self.interact()

        velocities = torch.stack(memory['velocities'])
        predicted_velocities = torch.stack(memory['predicted_velocities'])
        loss = self.calculate_vel_loss(velocities,predicted_velocities)
        self.opt.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), MAX_GRAD_NORM)
        

        # find a way to accumulate all gradients
        gradients = [param.grad for param in self.local_model.parameters()]
        return gradients, memory['timesteps']
    def run(self):
        self.train_cycles += 1
        # gradients, timesteps = self.train()
        gradients, timesteps = self.train()

        gradients_weighted = [grad * timesteps if grad is not None else None for grad in gradients]
        # gradients_weighted = [grad * timesteps for grad in gradients]
        # with Worker.save_lock:
        return gradients_weighted, timesteps

