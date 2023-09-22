#Import all libraries we will use
import random
import numpy as np
import cv2

#let's create a 80 x 80  (0-79)
#matrix with all pixels in black color
img = np.zeros((80,80),np.uint8)


def draw_odds(img,n,tic):
  tmp_img=img
  offset = (n * 5) + 1
  y = 0
  x = offset
  
  while tic > 0 :
    if x-offset >=4:
      y = y + 1
      x = offset
    
    print(x,y,n,tic,offset -x)
    tic = tic - 1
    tmp_img[x,y] = 255
    x = x + 1
    
  return tmp_img
############################
img = draw_odds(img,0,19)
img = draw_odds(img,1,31)
img = draw_odds(img,2,22)
img = draw_odds(img,3,17)
img = draw_odds(img,4,41)
img = draw_odds(img,5,50)
img = draw_odds(img,6,1)
img = draw_odds(img,7,5)
img = draw_odds(img,8,4)
img = draw_odds(img,9,65)
img = draw_odds(img,10,72)
img = draw_odds(img,11,83)
img = draw_odds(img,12,10)
img = draw_odds(img,13,45)
img = draw_odds(img,14,130)
img = draw_odds(img,15,143)

#img=draw_odds2(img,0,0)

cv2.imwrite("80_x_80_x_1.png",img)

#for i in [0,1,2,3,4,5,6,7,8,9,10] :
#  print (i,i%4)



