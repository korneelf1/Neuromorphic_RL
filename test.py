from environments import SimpleDrone_Discrete
from actor_critics import ActorCritic_ANN
import torch
import torch.nn.functional as F

env = SimpleDrone_Discrete(dt=0.05)
state_size = env.observation_space.shape[0]
action_size = env.action_space
model = ActorCritic_ANN(state_size, action_size)
torch.load('A3C/past_trainings/Figures/drone_snn.pt')

state, _ = env.reset()
done = False
while not done:
    # state to tensor
    try:
        state = torch.from_numpy(state)
    except:
        state = state
    # get network outputs on given state
    value, policy = model(state.unsqueeze(0))

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
    env.render()

            