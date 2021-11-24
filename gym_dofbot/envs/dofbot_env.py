import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import pybullet as p
import pybullet_data
import math
import numpy as np


import random


MAX_EPISODE_LEN = 20*100

class DofbotEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, DIRECT=False):
        self.step_counter = 0
        if not DIRECT:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT) # DIRECT mode, useful when the simulation device does not have a GPU
        # p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])
        self.action_space = spaces.Box(np.array([-1]*5), np.array([1]*5))
        self.observation_space = spaces.Box(np.array([-1]*5), np.array([1]*5))

    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        
        # perform the action
        for i in range(5):
            p.resetJointState(self.armUid,i, action[i])
        
        p.stepSimulation()
        state_dofbot = p.getLinkState(self.armUid, 4)[0]
        state_object, _ = p.getBasePositionAndOrientation(self.objectUid)
        
        self.camera_pos = p.getLinkState(self.armUid, 5)[0]
        self.cameraTargetPosition = p.getLinkState(self.armUid, 6)[0]
        # self.camera_bearing = p.getLinkState(self.armUid, 3)[1]


        if state_object[2]>0.45:
            reward = 1
            done = True
        else:
            reward = 0
            done = False

        self.step_counter += 1

        if self.step_counter > MAX_EPISODE_LEN:
            reward = 0
            done = True

        info = {'object_position': state_object}
        self.observation = state_dofbot
        return np.array(self.observation).astype(np.float32), reward, done, info
    
    
    def calcInverseKinematics(self, jointId, targetPosition):
        # orientation is optional
        orientation = p.getQuaternionFromEuler([-3.14159 / 2, 0, 0])
        # calculate the desired joint position
        targetPositionJoints = p.calculateInverseKinematics(self.armUid, jointId, targetPosition)
        
        return targetPositionJoints
    
    
    def setJointControl(self, controlArray):
        p.setJointMotorControlArray(self.armUid, range(5), p.POSITION_CONTROL, targetPositions=controlArray)
    
    def getPosition(self, jointID):
        return p.getLinkState(self.armUid, jointID)[0]

    def reset(self):
        self.step_counter = 0
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
        p.setGravity(0,0,-9.8)
        
        # get current path
        urdfRootPath=pybullet_data.getDataPath()
        
        # load plane URDF
        planeUid = p.loadURDF(os.path.join(urdfRootPath,"plane.urdf"), basePosition=[0,0,-0.65])
        # load table URDF
        tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"),basePosition=[0.5,0,-0.65])
        dofbot_path = os.path.join(os.path.dirname(__file__), 'arm.urdf')
        # load DOFBOT URDF
        self.armUid = p.loadURDF(dofbot_path, basePosition=[0,-0.2,0], useFixedBase=True)

        # change the appearance of DOFBOT parts
        p.changeVisualShape(self.armUid, -1, rgbaColor=[0,0,0,1])
        p.changeVisualShape(self.armUid, 0, rgbaColor=[0,1,0,1])
        p.changeVisualShape(self.armUid, 1, rgbaColor=[1,1,0,1])
        p.changeVisualShape(self.armUid, 2, rgbaColor=[0,1,0,1])
        p.changeVisualShape(self.armUid, 3, rgbaColor=[1,1,0,1])
        p.changeVisualShape(self.armUid, 4, rgbaColor=[0,0,0,1])
        p.changeVisualShape(self.armUid, 5, rgbaColor=[0,1,0,0])
        p.changeVisualShape(self.armUid, 6, rgbaColor=[0,0,1,0])
        p.changeVisualShape(self.armUid, 7, rgbaColor=[1,0,0,0.5])
        # reset pose of all DOFBOT joints
        rest_poses_dofbot = [0, 0, 0, 0, 0] # stay upright
        
        for i in range(5):
            p.resetJointState(self.armUid,i, rest_poses_dofbot[i])
        
        state_arm_0 = p.getLinkState(self.armUid, 0)[0]
        state_arm_1 = p.getLinkState(self.armUid, 1)[0]
        state_arm_2 = p.getLinkState(self.armUid, 2)[0]
        state_arm_3 = p.getLinkState(self.armUid, 3)[0]
        state_arm_4 = p.getLinkState(self.armUid, 4)[0]
        
        print("Link 0 position: ", state_arm_0)
        print("Link 1 position: ", state_arm_1)
        print("Link 2 position: ", state_arm_2)
        print("Link 3 position: ", state_arm_3)
        print("Link 4 position: ", state_arm_4)
        
        self.camera_pos = p.getLinkState(self.armUid, 5)[0]
        self.cameraTargetPosition = p.getLinkState(self.armUid, 6)[0]
        # self.camera_bearing = p.getLinkState(self.armUid, 3)[1]
        
        # randomly place the object (somewhere near the DOFBOT)
        state_object= [random.uniform(0.8,1.2),random.uniform(-0.1,0.3),0.05]
        self.objectUid = p.loadURDF(os.path.join(urdfRootPath, "random_urdfs/000/000.urdf"), basePosition=state_object)
        

        state_dofbot = p.getLinkState(self.armUid, 4)[0]
        
        
        self.observation = state_dofbot
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        
        
        return np.array(self.observation).astype(np.float32)


    def render(self, mode='human'):
        view_matrix = p.computeViewMatrix(cameraEyePosition=self.camera_pos,
                                                            cameraTargetPosition=self.cameraTargetPosition,
                                                            cameraUpVector=[0, 0, 1])
        proj_matrix = p.computeProjectionMatrixFOV(fov=45,
                                                     aspect=float(224) /224,
                                                     nearVal=0.1,
                                                     farVal=100.0)
        
        (_, _, px, _, _) = p.getCameraImage(width=224,
                                              height=224,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (224,224, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _get_state(self):
        return self.observation

    def close(self):
        p.disconnect()
