import gymnasium as gym
import random
from collections import namedtuple, deque


import torch


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# BATCH_SIZE 
class ReplayBuffer_Spiking(object):

    def __init__(self, capacity, interaction_max_length=200, padding='end', tbptt_length=10):
        '''
        Padding can be end or start. 
        End padding will padd the end of the sequence with zeros
        Start padding will pad the start of the sequence with zeros
        tbptt_length is the truncated backpropagation through time length, which defines the length of the sequences being stored. if None, the full sequence is stored'''
        self.memory = deque([], maxlen=capacity)
        self.max_length = interaction_max_length
        self.padding = padding
        self.tbptt_length = tbptt_length

    def push(self, state, action, next_state, reward):
        """Save a transition"""
        states = torch.zeros((self.max_length, 1))
        actions = torch.zeros((self.max_length, 1))
        next_states = torch.zeros((self.max_length, 1))
        rewards = torch.zeros((self.max_length))
        
        #states.fill_(float('inf'))
        #next_states.fill_(float('inf'))

        if state.shape[0] > self.max_length \
            or next_state.shape[0] > self.max_length \
            or action.shape[0] > self.max_length\
            or reward.shape[0] > self.max_length:
            print(state, action, next_state, reward)
            raise ValueError('Interaction length exceeds maximum length')
        else:
            if self.padding == 'end':
                states[:state.shape[0]] = state
                actions[:action.shape[0]] = action
                next_states[:next_state.shape[0]] = next_state
                rewards[:reward.shape[0]] = reward
            elif self.padding == 'start':
                states[self.max_length - state.shape[0]:] = state
                action[self.max_length - state.shape[0]:] = action
                next_states[self.max_length - state.shape[0]:] = next_state
                rewards[self.max_length - state.shape[0]:] = reward
            else:
                raise ValueError('Padding must be either start or end')
        if self.tbptt_length is not None:
            for i in range(0, self.max_length, self.tbptt_length):
                if torch.all(torch.all(states[i:] == 0, dim=0)==1):
                # if torch.isinf(next_state_batch).all(dim=-1):
                    break
                self.memory.append(Transition(states[i:i+self.tbptt_length], actions[i:i+self.tbptt_length], next_states[i:i+self.tbptt_length], rewards[i:i+self.tbptt_length]))
        else:
            self.memory.append(Transition(states, actions, next_states, rewards))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

