import gym
import gym_dofbot
import time
import math


env = gym.make('dofbot-v0')

env.reset()
env.render()
counter = 0
# while 1:
#     env.render()
#     time.sleep(0.01)
#     counter += 1
    
#     env.step([-math.pi / 4 + 0.25 * math.pi * math.sin(counter / 100), -math.pi/3, -math.pi/4, -math.pi/8, 0])

print("gripper point: ", env.getPosition(7))

endPosition = env.getPosition(0)

targetPosition = [endPosition[0] , endPosition[1] + 0.2, endPosition[2] + 0.1]

targetPositionJoints = env.calcInverseKinematics(7, targetPosition)


# env.step([targetPositionJoints[0], targetPositionJoints[1], targetPositionJoints[2], targetPositionJoints[3], targetPositionJoints[4]])



counter = 25



while 1:
    env.render()
    targetPosition = [endPosition[0] + 0.3 * math.sin(counter / 50), endPosition[1] + 0.2, endPosition[2] + 0.01]
    targetPositionJoints = env.calcInverseKinematics(7, targetPosition)
    print(targetPositionJoints)
    targetPositionJoints = [0, 0, 0, 0, 0]
    env.step([targetPositionJoints[0], targetPositionJoints[1], targetPositionJoints[2], targetPositionJoints[3], targetPositionJoints[4]])
    time.sleep(0.01)
    counter += 1
    

env.render()
