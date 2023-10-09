import gymnasium as gym
import numpy as np
import cv2
#import gym_bnlbot

def resize_frame(frame):
   # frame = frame[30:-12,5:-4]
   
    #print ('frame type', type(frame))
    #print ('frame', frame)
   
    #frame = np.average(frame,axis = 2)
    frame = cv2.resize(frame,(84,84),interpolation = cv2.INTER_NEAREST)
    frame = np.array(frame,dtype = np.uint8)
    return frame


def initialize_new_race(name, env, agent):
    """We don't want an agents past game influencing its new game, so we add in some dummy data to initialize"""
    
    env.reset()
    starting_frame = resize_frame(env.step(0)[0])

    dummy_action = 0
    dummy_reward = 0
    dummy_done = False
    for i in range(3):
        agent.memory.add_experience(starting_frame, dummy_reward, dummy_action, dummy_done)

def make_env(name, agent, debug):
    
    if debug :    
        env = gym.make(name, render_mode="human")
        env.metadata['render_fps'] = 5
    else:
        env = gym.make(name)
    return env

def take_step(name, env, agent, score, debug):
    
    #1 and 2: Update timesteps and save weights
    agent.total_timesteps += 1
    if agent.total_timesteps % 5000 == 0:
      agent.model.save_weights(agent.weight_filename)
      print('\nWeights saved!')

    #3: Take action
    #bnl add next_frame_trunc
    next_frame, next_frames_reward, next_frame_terminal, next_frame_trunc, info = env.step(agent.memory.actions[-1])
    next_frame_terminal = next_frame_terminal or next_frame_trunc
    #4: Get next state
    next_frame = resize_frame(next_frame)
    new_state = [agent.memory.frames[-3], agent.memory.frames[-2], agent.memory.frames[-1], next_frame]
    new_state = np.moveaxis(new_state,0,2)/255 #We have to do this to get it into keras's goofy format of [batch_size,rows,columns,channels]
    new_state = np.expand_dims(new_state,0) #^^^
    
    #5: Get next action, using next state
    next_action = agent.get_action(new_state)

    #7: Now we add the next experience to memory
    agent.memory.add_experience(next_frame, next_frames_reward, next_action, next_frame_terminal)

    #6: If game is over, return the score
    if next_frame_terminal:
        return (score + next_frames_reward),True


    #8: If we are trying to debug this then render
    if debug:
        env.render()

    #9: If the threshold memory is satisfied, make the agent learn from memory
    if len(agent.memory.frames) > agent.starting_mem_len:
        agent.learn(debug)

    return (score + next_frames_reward),False

def play_episode(name, env, agent, debug):
    initialize_new_race(name, env, agent)
    done = False
    score = 0
    while not done:
        score,done = take_step(name,env,agent,score, debug)
    return score

