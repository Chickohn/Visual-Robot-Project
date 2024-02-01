import os
import gymnasium as gym
import panda_gym
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.logger import configure
from time import sleep
from typing import Any, Dict, Tuple, Optional
from scipy.spatial.transform import Rotation as R
from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.core import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import angle_distance
import pybullet as p

# Create the environment
env = gym.make('PandaReach-v3', render_mode="human")
print("Environment:", env)

# Configure logger for TensorBoard
log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)
logger = configure(log_dir, ["stdout", "tensorboard"])

# Initialize the RL agent
model = DDPG("MultiInputPolicy", env, learning_rate=0.01, batch_size=32, gamma=0.99, verbose=1)

# Train the agent
model.learn(total_timesteps=1000)  # Number of timesteps to train for

model_path = "./trained_models/trained_reach"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
model.save(model_path)

# Test the trained agent
observation, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()