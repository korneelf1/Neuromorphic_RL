import numpy as np
import gym

from gym.wrappers import TimeLimit
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from environments import SimpleDrone
import matplotlib.pyplot as plt

hyperparams = {
    'n_envs': 8,
    'n_timesteps': 2e6,
    'policy': 'MlpPolicy',
    'n_steps': 2048,
    'nminibatches': 32,
    'lam': 0.95,
    'gamma': 0.99,
    'noptepochs': 10,
    'ent_coef': 0.0,
    'learning_rate': 3e-4,
    'cliprange': 0.2}

# Create the environment
env_id = "Pendulum-v1"
env = make_vec_env(env_id, n_envs=8)
# Normalize
env = VecNormalize(env, gamma=0.9)

# Create the evaluation env (could be used in `EvalCallback`)
eval_env = make_vec_env(env_id, n_envs=1)
eval_env = VecNormalize(eval_env, gamma=0.9, training=False, norm_reward=False)

# Instantiate the agent
model = A2C(
        "MlpPolicy",
        env,
        gamma=0.98,
        # Using https://proceedings.mlr.press/v164/raffin22a.html
        use_sde=True,
        sde_sample_freq=4,
        learning_rate=1e-3,
        verbose=1,
)

# Train the agent
model.learn(total_timesteps=int(1e5))

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)

print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Create a function to instantiate the environment
def make_env():
        env = TimeLimit(SimpleDrone(render_mode='human'), max_episode_steps=200)
        return env

# Create the vectorized environment
env = SubprocVecEnv([make_env for _ in range(hyperparams['n_envs'])])

# Normalize the environment
env = VecNormalize(env, gamma=hyperparams['gamma'])

# Instantiate the agent
model = A2C(hyperparams['policy'], env, verbose=1, tensorboard_log="./tb/")

# Train the agent
model.learn(total_timesteps=int(hyperparams['n_timesteps']))

# Save the agent
model.save("a2c_simple_drone")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)

print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)

# Render the environment of the evaluation
obs = eval_env.reset()
for i in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = eval_env.step(action)
    eval_env.render()
    if done:
        obs = eval_env.reset()
