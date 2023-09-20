print('init 1 start')

import gymnasium 
from gymnasium.envs.registration import register

register(
    id='bnlbot-v0',
    entry_point='gym_bnlbot.envs:Bnlbot',
)

print('init 1 stop')
