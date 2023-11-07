import gym
from stable_baselines3 import A2C
import tensorflow as tf
import gym
import torch
from stable_baselines3 import A2C
from torch import nn

env = gym.make("CartPole-v1")
policy_kwargs = dict(
  net_arch=[11],shared_layers=1)
model = A2C("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs)
model.learn(total_timesteps=1000000)

# model.save("a2c_cartpole")
class PyTorchModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 11) 
            self.fc2 = nn.Linear(11, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

def convert_to_pytorch():
    # Load SB3 model
    env = gym.make("CartPole-v1")
    model = A2C.load("a2c_cartpole") 

    # Get weights as NumPy arrays
    weights = model.get_parameters()

    # Define PyTorch model with same architecture
    # model = PyTorchModel()


    # Load weights into PyTorch model
    # for p, w in zip(model.parameters(), weights):
    for p in weights['policy']:
        print(weights['policy'][p])
        # p.data = torch.from_numpy(w)

    # Model is now loaded with weights from SB3 model  
    torch.save(model.state_dict(), "pytorch_cartpole.pt")

convert_to_pytorch()