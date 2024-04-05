import random
from collections import namedtuple, deque
# from ..CartPole_modified import CartPole_fake

import torch



import random
from collections import deque, namedtuple
######################################################################
# Replay Memory
# -------------
#
# We'll be using experience replay memory for training our DQN. It stores
# the transitions that the agent observes, allowing us to reuse this data
# later. By sampling from it randomly, the transitions that build up a
# batch are decorrelated. It has been shown that this greatly stabilizes
# and improves the DQN training procedure.
#
# For this, we're going to need two classes:
#
# -  ``Transition`` - a named tuple representing a single transition in
#    our environment. It essentially maps (state, action) pairs
#    to their (next_state, reward) result, with the state being the
#    screen difference image as described later on.
# -  ``ReplayMemory`` - a cyclic buffer of bounded size that holds the
#    transitions observed recently. It also implements a ``.sample()``
#    method for selecting a random batch of transitions for training.
#

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
Transition_Spiking = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'hidden_states'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class ReplayMemory_Full_Seq(object):

    def __init__(self, capacity, interaction_max_length=200, padding='end', tbptt_length=10, keep_hidden_states=False, hidden_states_shape=(2,128)):
        '''
        Padding can be end or start. 
        End padding will padd the end of the sequence with zeros
        Start padding will pad the start of the sequence with zeros
        tbptt_length is the truncated backpropagation through time length, which defines the length of the sequences being stored. if None, the full sequence is stored'''
        self.memory = deque([], maxlen=capacity)
        self.max_length = interaction_max_length
        self.padding = padding
        self.tbptt_length = tbptt_length
        self.keep_hidden_states = keep_hidden_states
        self.hidden_shape = hidden_states_shape

    def push(self, state, action, next_state, reward, hidden_state=None):
        """Save a transition"""
        states = torch.zeros((self.max_length, 4))
        actions = torch.zeros((self.max_length, 1))
        next_states = torch.zeros((self.max_length, 4))
        rewards = torch.zeros((self.max_length))
        if self.keep_hidden_states:
            hidden_states = torch.zeros((self.max_length, *self.hidden_shape))
        
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
                if self.keep_hidden_states:
                    hidden_states[:hidden_state.shape[0]] = hidden_state

            elif self.padding == 'start':
                states[self.max_length - state.shape[0]:] = state
                actions[self.max_length - state.shape[0]:] = action
                next_states[self.max_length - state.shape[0]:] = next_state
                rewards[self.max_length - state.shape[0]:] = reward
            else:
                raise ValueError('Padding must be either start or end')
        if self.tbptt_length is not None:
            for i in range(0, self.max_length, self.tbptt_length):
                if torch.all(torch.all(states[i:] == 0, dim=0)==1):
                # if torch.isinf(next_state_batch).all(dim=-1):
                    break
                if self.keep_hidden_states:
                    self.memory.append(Transition_Spiking(states[i:i+self.tbptt_length], actions[i:i+self.tbptt_length], next_states[i:i+self.tbptt_length], rewards[i:i+self.tbptt_length], hidden_states[i:i+self.tbptt_length]))
                else:
                    self.memory.append(Transition(states[i:i+self.tbptt_length], actions[i:i+self.tbptt_length], next_states[i:i+self.tbptt_length], rewards[i:i+self.tbptt_length]))
        else:
            if SPIKING:
                self.memory.append(Transition_Spiking(states, actions, next_states, rewards, hidden_states))
            else:
                self.memory.append(Transition(states, actions, next_states, rewards))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

