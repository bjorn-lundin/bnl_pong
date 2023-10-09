import the_agent
import environment
import matplotlib.pyplot as plt
import time
from collections import deque
import numpy as np
from gym_bnlbot.envs.bnlbot import Bnlbot
import gym_bnlbot
import os


#name = 'PongDeterministic-v4'
name= 'bnlbot-v0'
#debug=True
debug=False
#set debug to true for rendering

agent = the_agent.Agent(possible_actions=[Bnlbot.DO_NOT_PLACE_BET,Bnlbot.DO_PLACE_BET],starting_mem_len=50000,max_mem_len=750000,starting_epsilon = 1, learn_rate = .00025)
env = environment.make_env(name,agent,debug)

last_100_avg = [-21]
scores = deque(maxlen = 100)
max_score = -21

#""" If testing:
if os.path.isfile(agent.weight_filename):
  agent.model.load_weights(agent.weight_filename)
  agent.model_target.load_weights(agent.weight_filename)
  agent.epsilon = 0.0
#"""

env.reset()

for i in range(1000000):
    timesteps = agent.total_timesteps
    timee = time.time()
    score = environment.play_episode(name, env, agent, debug) 
    scores.append(score)
    if score > max_score:
        max_score = score

    print('\nEpisode: ' + str(i))
    print('Steps: ' + str(agent.total_timesteps - timesteps))
    print('Duration: ' + str(time.time() - timee))
    print('Score: ' + str(score))
    print('Max Score: ' + str(max_score))
    print('Epsilon: ' + str(agent.epsilon))

    if i%10==0 and i!=0:
        last_100_avg.append(sum(scores)/len(scores))
#        plt.plot(np.arange(0,i+1,100),last_100_avg)
#        plt.show()
        print('last_100_avg',last_100_avg)

