#import gymnasium as gym
print('init 2 start')

from gymnasium.envs.registration import register


from gym_bnlbot.envs.bnlbot import Bnlbot

register(
     id="gym_bnlbot/Bnlbot-v0",
     entry_point="gym_bnlbot.envs:Bnlbot",
     max_episode_steps=3000,
)
print('init 2 stop')

