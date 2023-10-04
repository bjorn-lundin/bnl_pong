from gym_bnlbot.envs.bnlbot import Bnlbot
import gymnasium as gym
import numpy as np
import cv2
import gym_bnlbot


def initialize_new_race(name, env, agent):
    env.reset()

def make_env(name, agent, debug):
    
    if debug :    
        env = gym.make(name, render_mode="rgb")
        env.metadata['render_fps'] = 5
    else:
        env = gym.make(name)
    return env

def take_step(name, env, agent, score, debug):
    
    #bnl add next_frame_trunc
    next_frame, next_frames_reward, next_frame_terminal, next_frame_trunc, info = env.step(Bnlbot.DO_PLACE_BET)
    next_frame_terminal = next_frame_terminal or next_frame_trunc
    #5: Get next action, using next state
    next_action = env.action_space.sample()

    #6: If game is over, return the score
    if next_frame_terminal:
        if agent is not None :
          agent.memory.add_experience(next_frame, next_frames_reward, next_action, next_frame_terminal)
        return (score + next_frames_reward),True

    return (score + next_frames_reward),False

def play_episode(name, env, agent, debug):
    initialize_new_race(name, env, agent)
    done = False
    score = 0
    while True:
        score,done = take_step(name,env,agent,score, debug)
        if done:
            break
    return score

name= 'bnlbot-v0'
debug=True
agent=None
#debug=False
#set debug to true for rendering
env = make_env(name,agent,debug)
#env.reset()

#while True:
#  score = play_episode(name, env, agent, debug) 

score = play_episode(name, env, agent, debug) 

