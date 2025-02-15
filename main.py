import pybullet as p
import pybullet_data
import time
from src.simulation import Simulation
from src.robot import Robot
from src.dynamixel import Dynamixel
import numpy as np
import time

# p.connect(p.GUI)
# p.resetSimulation()
# p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.setAdditionalSearchPath("/Users/alexander/Developer/Robot-Manipulator/")
# p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
# p.setGravity(0, 0, -10)
# p.setPhysicsEngineParameter(solverResidualThreshold=0)


# # Generate panda, plane, other objects
# orientation = p.getQuaternionFromEuler([0, 0, 0])
# # panda
# robot = p.loadURDF(
#     "simulation/model/robot.urdf",
#     useFixedBase=True,
#     globalScaling=0.01
# )

# p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
# joint1 = p.addUserDebugParameter("joint1", -3.14, 3.14, 0)
# joint2 = p.addUserDebugParameter("joint2", -3.14, 3.14, 0)
# joint3 = p.addUserDebugParameter("joint3", -3.14, 3.14, 0)
# joint4 = p.addUserDebugParameter("joint4", -3.14, 3.14, 0)
# joint5 = p.addUserDebugParameter("joint5", -3.14, 3.14, 3.14)
# joint6 = p.addUserDebugParameter("joint6", -3.14, 3.14, 0)

# x_param = p.addUserDebugParameter("x", -1, 1, 0)
# y_param = p.addUserDebugParameter("y", -1, 1, 0)
# z_param = p.addUserDebugParameter("z", -1, 1, 0)
# joint_num = p.getNumJoints(robot)
# ee_index = joint_num - 1

# joint_positions = [0, 0, 0, 0, 3.14, 0]
# index = 0
# for j in range(p.getNumJoints(robot)):
#     p.changeDynamics(robot, j, linearDamping=0, angularDamping=0)
#     info = p.getJointInfo(robot, j)
#     joint_type = info[2]
#     if joint_type == p.JOINT_PRISMATIC or joint_type == p.JOINT_REVOLUTE:
#         p.resetJointState(robot, j, joint_positions[index])
#         index = index + 1
        
# ee_position = p.getLinkState(robot, ee_index)[0]


follower_dynamixel = Dynamixel.Config(baudrate=1_000_000, device_name="/dev/tty.usbmodem58FA0959341").instantiate()
follower = Robot(follower_dynamixel, servo_ids=[1, 2, 3, 4, 5, 6])

s = Simulation()
while True:
    new_joint_positions = s.step()


    follower.set_goal_pos(new_joint_positions)
        
follower._disable_torque()

# p.disconnect()
