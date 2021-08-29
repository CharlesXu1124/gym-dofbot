import gym
import gym_dofbot
import time
import math

env = gym.make('dofbot-v0')

env.reset()
env.render()
counter = 0
while 1:
    env.render()
    time.sleep(0.01)
    counter += 1
    
    env.step([-math.pi / 4 + 0.25 * math.pi * math.sin(counter / 100), -math.pi/3, -math.pi/4, -math.pi/8, 0])


env.render()