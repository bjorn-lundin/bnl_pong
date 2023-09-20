Using Tensorflow to play Pong
=============================

This is a fork from Meredevs/DQN_Pong
with adjustments for migration gym -> gymnasium

After 24 hours with an iMac 27" running Ubuntu 22.04
the built-in AI gets beat by 10 points on avg (21-11)


Clone it
    cd 
    cd git
    git clone https://github.com/bjorn-lundin/bnl_pong

Install this
    sudo apt install python3-venv -y
    sudo apt install python3-tk
    python3 -m venv bnl_pong
    source bnl_pong/bin/activate
    cd bnl_pong/
    pip install numpy
    pip install gymnasium[atari]
    pip install gymnasium[accept-rom-license]
    pip install matplotlib
    pip install tensorflow==2.9
    pip install opencv-python
    python pong_orig.py --render --resume



32 Gb RAM but with tensorflow 2.9 it does not use more that 8-9 GB

Tensorflow 2.13 eats everything. 

Must be a mem-leak

    bnl_pong) bnl@iMac:~/git/bnl_pong$ lscpu
    Architecture:            x86_64
      CPU op-mode(s):        32-bit, 64-bit
      Address sizes:         36 bits physical, 48 bits virtual
      Byte Order:            Little Endian
    CPU(s):                  8
      On-line CPU(s) list:   0-7
    Vendor ID:               GenuineIntel
      Model name:            Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz
        CPU family:          6
        Model:               58
        Thread(s) per core:  2
        Core(s) per socket:  4
        Socket(s):           1
        Stepping:            9

#The result

http://lundin.duckdns.org/pong.html


You also need to get the customized gym-class registered

    (bnl_pong) bnl@iMac:~/git/bnl_pong$ cd gym-bnlbot
    (bnl_pong) bnl@iMac:~/git/bnl_pong/gym-bnlbot$ ls
    gym_bnlbot  setup.py

    (bnl_pong) bnl@iMac:~/git/bnl_pong/gym-bnlbot$ pip install -e .
    Obtaining file:///home/bnl/git/bnl_pong/gym-bnlbot
      Preparing metadata (setup.py) ... done
    Installing collected packages: Bnlbot
      Running setup.py develop for Bnlbot
    Successfully installed Bnlbot-0.0.1
