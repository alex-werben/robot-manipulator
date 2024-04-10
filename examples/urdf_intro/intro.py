import pybullet as p
import time
import pybullet_data
import pybullet_robots


physicsClient = p.connect(p.GUI)  #or p.DIRECT for non-graphical version

p.setAdditionalSearchPath("/Users/alexander/Developer/Robot-Manipulator/experiments")

startPos = [0, 0, 0]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
robot_id = p.loadURDF("09.urdf", startPos)
joint_index = 9

for i in range (1000):
    p.stepSimulation()
    time.sleep(1./240.)


p.disconnect()
