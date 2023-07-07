import torch
import helper_functions as hf
import gym

env = gym.make("MountainCar-v0", render_mode = 'human')

actor = hf.Feedforward(2,3,3,120)
actor.load_state_dict(torch.load('actor_500.txt'))
actor.eval()

state, info = env.reset()

terminal = False
while not terminal:
    state = torch.from_numpy(state)
    action = actor(state.unsqueeze(0)).argmax()

    state, _, terminal, _, _ = env.step(int(action))
    print(action)
env.close()