import gymnasium as gym
import panda_gym
import numpy as np
from stable_baselines3 import DDPG
from time import sleep
from typing import Any, Dict, Tuple
from typing import Optional
import numpy as np
from scipy.spatial.transform import Rotation as R
from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.core import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import angle_distance
import pybullet as p

# Create the environment
env = gym.make('PandaFlipDense-v3', render_mode="human")
# env = CustomFlipEnv(render_mode="human") 
print("Environment:", env)

# Initialize the RL agent
model = DDPG("MultiInputPolicy", env, learning_rate=0.01, batch_size=64, gamma=0.99, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)  # Number of timesteps to train for

# Test the trained agent
observation, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

#recreate flip task. Improve rewards. Stop sparse rewards -> make dense as they get closer to the goal. ie touching cube, grasping cube, orientating cube closer to the goal...
# tensor board, github, kuka camera