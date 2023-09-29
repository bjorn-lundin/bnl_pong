print('bnlbot start')
import numpy as np
import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
#from gym import error, spaces, utils
#from gym.utils import seeding
import os
from pathlib import Path
import copy

RACEFILE_DIRECTORY = os.environ.get('BOT_HISTORY') + '/data/ai/pong/1st/back/win/train'

print('bnlbot stop')

  ##########################################


class Bnlbot(gym.Env):
  metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}  
  DO_NOT_PLACE_BET=0
  DO_PLACE_BET=1

  ODDS_TO_TIC_DICT = {
       "0.0" :  0,
       "1.01":  1,   "1.02":   2,  "1.03":   3,  "1.04":   4,   "1.05":  5, "1.06": 6, "1.07": 7, "1.08": 8, "1.09": 9, 
       "1.1":  10,   "1.11":  11,  "1.12":  12,  "1.13":  13,   "1.14": 14, "1.15": 15, "1.16": 16, "1.17": 17, "1.18": 18, "1.19": 19, 
       "1.2":  20,   "1.21":  21,  "1.22":  22,  "1.23":  23,   "1.24": 24, "1.25": 25, "1.26": 26, "1.27": 27, "1.28": 28, "1.29": 29, 
       "1.3":  30,   "1.31":  31,  "1.32":  32,  "1.33":  33,   "1.34": 34, "1.35": 35, "1.36": 36, "1.37": 37, "1.38": 38, "1.39": 39, 
       "1.4":  40,   "1.41":  41,  "1.42":  42,  "1.43":  43,   "1.44": 44, "1.45": 45, "1.46": 46, "1.47": 47, "1.48": 48, "1.49": 49, 
       "1.5":  50,   "1.51":  51,  "1.52":  52,  "1.53":  53,   "1.54": 54, "1.55": 55, "1.56": 56, "1.57": 57, "1.58": 58, "1.59": 59, 
       "1.6":  60,   "1.61":  61,  "1.62":  62,  "1.63":  63,   "1.64": 64, "1.65": 65, "1.66": 66, "1.67": 67, "1.68": 68, "1.69": 69, 
       "1.7":  70,   "1.71":  71,  "1.72":  72,  "1.73":  73,   "1.74": 74, "1.75": 75, "1.76": 76, "1.77": 77, "1.78": 78, "1.79": 79, 
       "1.8":  80,   "1.81":  81,  "1.82":  82,  "1.83":  83,   "1.84": 84, "1.85": 85, "1.86": 86, "1.87": 87, "1.88": 88, "1.89": 89, 
       "1.9":  90,   "1.91":  91,  "1.92":  92,  "1.93":  93,   "1.94": 94, "1.95": 95, "1.96": 96, "1.97": 97, "1.98": 98, "1.99": 99, 
       "2.0": 100,   "2.02": 101,  "2.04": 102,  "2.06": 103,   "2.08": 104, "2.1": 105, "2.12": 106, "2.14": 107, "2.16": 108, "2.18": 109, 
       "2.2": 110,   "2.22": 111,  "2.24": 112,  "2.26": 113,   "2.28": 114, "2.3": 115, "2.32": 116, "2.34": 117, "2.36": 118, "2.38": 119, 
       "2.4": 120,   "2.42": 121,  "2.44": 122,  "2.46": 123,   "2.48": 124, "2.5": 125, "2.52": 126, "2.54": 127, "2.56": 128, "2.58": 129, 
       "2.6": 130,   "2.62": 131,  "2.64": 132,  "2.66": 133,   "2.68": 134, "2.7": 135, "2.72": 136, "2.74": 137, "2.76": 138, "2.78": 139, 
       "2.8": 140,   "2.82": 141,  "2.84": 142,  "2.86": 143,   "2.88": 144, "2.9": 145, "2.92": 146, "2.94": 147, "2.96": 148, "2.98": 149, 
       "3.0": 150,   "3.05": 151,  "3.1":  152,  "3.15": 153,   "3.2" : 154, "3.25": 155, "3.3": 156, "3.35": 157, "3.4": 158, "3.45": 159, 
       "3.5": 160,   "3.55": 161,  "3.6":  162,  "3.65": 163,   "3.7" : 164, "3.75": 165, "3.8": 166, "3.85": 167, "3.9": 168, "3.95": 169, 
       "4.0": 170,   "4.1":  171,  "4.2":  172,  "4.3":  173,   "4.4" : 174, "4.5": 175, "4.6": 176, "4.7": 177, "4.8": 178, "4.9": 179, 
       "5.0": 180,   "5.1":  181,  "5.2":  182,  "5.3":  183,   "5.4" : 184, "5.5": 185, "5.6": 186, "5.7": 187, "5.8": 188, "5.9": 189, 
       "6.0": 190,   "6.2":  191,  "6.4":  192,  "6.6":  193,   "6.8" : 194, "7.0": 195, "7.2": 196, "7.4": 197, "7.6": 198, "7.8": 199, 
       "8.0": 200,   "8.2":  201,  "8.4":  202,  "8.6":  203,   "8.8" : 204, "9.0": 205, "9.2": 206, "9.4": 207, "9.6": 208, "9.8": 209, 
      "10.0": 210,  "10.5":  211,  "11.0": 212,  "11.5": 213,  "12.0" : 214, "12.5": 215, "13.0": 216, "13.5": 217, "14.0": 218, "14.5": 219, 
      "15.0": 220,  "15.5":  221,  "16.0": 222,  "16.5": 223,  "17.0" : 224, "17.5": 225, "18.0": 226, "18.5": 227, "19.0": 228, "19.5": 229, 
      "20.0": 230,  "21.0":  231,  "22.0": 232,  "23.0": 233,  "24.0" : 234, "25.0": 235, "26.0": 236, "27.0": 237, "28.0": 238, "29.0": 239, 
      "30.0": 240,  "32.0":  241,  "34.0": 242,  "36.0": 243,  "38.0" : 244, "40.0": 245, "42.0": 246, "44.0": 247, "46.0": 248, "48.0": 249, 
      "50.0": 250,  "55.0":  251,  "60.0": 252,  "65.0": 253,  "70.0" : 254, "75.0": 255, "80.0": 256, "85.0": 257, "90.0": 258, "95.0": 259, 
     "100.0": 260, "110.0":  261, "120.0": 262, "130.0": 263, "140.0" : 264, "150.0": 265, "160.0": 266, "170.0": 267, "180.0": 268, "190.0": 269, 
     "200.0": 270, "210.0":  271, "220.0": 272, "230.0": 273, "240.0" : 274, "250.0": 275, "260.0": 276, "270.0": 277, "280.0": 278, "290.0": 279, 
     "300.0": 280, "310.0":  281, "320.0": 282, "330.0": 283, "340.0" : 284, "350.0": 285, "360.0": 286, "370.0": 287, "380.0": 288, "390.0": 289, 
     "400.0": 290, "410.0":  291, "420.0": 292, "430.0": 293, "440.0" : 294, "450.0": 295, "460.0": 296, "470.0": 297, "480.0": 298, "490.0": 299, 
     "500.0": 300, "510.0":  301, "520.0": 302, "530.0": 303, "540.0" : 304, "550.0": 305, "560.0": 306, "570.0": 307, "580.0": 308, "590.0": 309, 
     "600.0": 310, "610.0":  311, "620.0": 312, "630.0": 313, "640.0" : 314, "650.0": 315, "660.0": 316, "670.0": 317, "680.0": 318, "690.0": 319, 
     "700.0": 320, "710.0":  321, "720.0": 322, "730.0": 323, "740.0" : 324, "750.0": 325, "760.0": 326, "770.0": 327, "780.0": 328, "790.0": 329, 
     "800.0": 330, "810.0":  331, "820.0": 332, "830.0": 333, "840.0" : 334, "850.0": 335, "860.0": 336, "870.0": 337, "880.0": 338, "890.0": 339, 
     "900.0": 340, "910.0":  341, "920.0": 342, "930.0": 343, "940.0" : 344, "950.0": 345, "960.0": 346, "970.0": 347, "980.0": 348, "990.0": 349, 
    "1000.0": 350 }




  ##########################################
  def __init__(self, render_mode=None, size=16):
    print('init')
    # stuff from https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
    
    self.size = size  # Num runners    
    
    self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(0, size - 1, shape=(1,), dtype=int)
               # "agent": spaces.Box(0, size - 1, shape=(1,), dtype=float),
               # "target": spaces.Box(0, size - 1, shape=(1,), dtype=float),
            }
        )    
    
    
    # We have 2 actions, corresponding to "dont place bet", "place bet on leader"
    #                                         0                     1
    self.action_space = spaces.Discrete(2)   
    
    assert render_mode is None or render_mode in self.metadata["render_modes"]
    self.render_mode = render_mode    
    self.window = None
    self.clock = None    
    
    #bnl stuff below
    
    #set up data structure
    self.racefile_idx = -1
    self.filehandle = None
    self.filename = None
    self.racefile_list = []
    self.line_number = -1

    dir_list = os.listdir(RACEFILE_DIRECTORY)
    for dirname in dir_list:
        mydir = Path(RACEFILE_DIRECTORY + '/' + dirname)
        #print(dirname + ' ' + str(mydir.is_dir()))
        if mydir.is_dir():
            file_list = os.listdir(RACEFILE_DIRECTORY + '/' + dirname)
            for filename in file_list:
                if filename == '.DS_Store' :
                    pass
                else :
                    self.racefile_list.append(dirname + '/' + filename)
                    #print(dirname + '/' + filename)

#    for i in self.racefile_list:
#        print(i)
#    print(len(self.racefile_list))
#    a=1/0

  ##########################################

  def get_observation(self):
    self.line = self.filehandle.readline().strip()
    self.line_number = self.line_number +1

    #print ('get_observation.self.line',self.line)
    data = self.line.split(',')
    #print ('get_observation.data', data)
    #print ('get_observation.data[17:32+1]',data[17:32+1])
    ll=[]
    for d in data[17:32+1] :
       ll.append(float(d))
    #print ('get_observation.ll',ll)
        
    ll2=[]
    for d in ll :
       ll2.append(Bnlbot.ODDS_TO_TIC_DICT[str(d)])
    #print ('get_observation.ll2',ll2)
    
    arr = np.array(ll2)
    print ('get_observation.arr',arr)
    
    return {'obs' : arr}
#    return arr 
  ##########################################
  def get_reward(self,idx):
    #print ('get_reward.self.line',self.line)
    data = self.line.split(',')
    #print ('get_reward.data[33:48+1]',data[33:48+1])
    ll=[]
    for d in data[33:48+1] :
       ll.append(float(d))
    print ('get_reward.ll', ll)
    print ('get_reward.idx', idx)
    print ('get_reward.ll[idx]',ll[idx], 'idx', idx)
    return ll[idx]
#      a=1/0
  ##########################################

  def step(self, action):
    terminated = False
    truncated = False
    rew = 0.0
    info = { "stuff": "no_info" }
    #print('step')
    #print('action ' + str(action))
    ob = self.get_observation()
    #check for eof
    if len(self.line) == 0 :
      truncated = True
    #print('step.ob', ob)
    #print('step.type(ob)', type(ob))

    if not truncated:

      action = Bnlbot.DO_PLACE_BET

      #decide to bet or not
      if action == Bnlbot.DO_PLACE_BET :
        #do bet on first runner found with lowest odds
        lowest = 10000.0
        idx = 0
        selidx = 0
        b=[]
        for odds in ob['obs']:
            if 0 < odds and odds < lowest :
                lowest = odds
                selidx = idx
                #print('step.newlowest', lowest)
            idx = idx +1

        if lowest > 1000 :
            print('did not find a valid odds')
            print('filename',self.filename)
            print('self.line_number',self.line_number)
            a=1/0

        if lowest >=250 :
            print('race has ended')
            truncated = True

        if not truncated:
          #selidx = selidx -1 ; # off by one - 0-15 not 1-16  
          #print('step.selidx', selidx)
          rew = self.get_reward(selidx)
          #print('step.rew', rew)
      else:
        pass

    else:
      pass

    return (ob,rew,terminated,truncated, info)

  ##########################################

  def reset(self):

    self.racefile_name = []
    self.racefile_idx = self.racefile_idx +1

    self.filename = RACEFILE_DIRECTORY + '/' + self.racefile_list[self.racefile_idx]
    self.filehandle = open(self.filename)
    
    print('new self.racefile_list_idx', self.racefile_idx)
    print('new self.filename', self.filename)

    return self.get_observation()


  ##########################################

  def render(self, mode='human', close=False):
    print('render')

  ##########################################
