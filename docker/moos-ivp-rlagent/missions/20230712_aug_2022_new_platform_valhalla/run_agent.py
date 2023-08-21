import config
import os
import sys
import gym
import gym_aquaticus
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from stable_baselines3 import PPO


log_dir = 'models/defend_easy2'


env = gym.make('gym_aquaticus:aquaticus-v0', sim_script='./launch_demo.sh', verbose=0, perpetual=1)
model = PPO.load(log_dir + '/best_model', custom_objects={"learning_rate":0.003,"lr_schedule":lambda _: 0.0, "clip_range": lambda _: 0.2},env=env)
obs = env.reset()
done_count = 0
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
        done_count += 1
        
env.close()
