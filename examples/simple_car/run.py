import pybullet as p
import pybullet_data
import time
PATH = "/Users/alexander/Developer/Robot-Manipulator/examples/"


def main():
  client = p.connect(p.GUI)

  p.setGravity(0, 0, -10, physicsClientId=client)
  wheel_indices = [1, 3, 4, 5]
  hinge_indices = [0, 2]
  angle = p.addUserDebugParameter('Steering', -0.5, 0.5, 0)
  throttle = p.addUserDebugParameter('Throttle', 0, 20, 0)

  p.setAdditionalSearchPath(pybullet_data.getDataPath())
  planeId = p.loadURDF("plane.urdf")
  p.setAdditionalSearchPath(PATH)
  car = p.loadURDF("simplecar.urdf", basePosition=[0, 0, 0.5])
  
  number_of_joints = p.getNumJoints(car)
  for joint_number in range(number_of_joints):
      info = p.getJointInfo(car, joint_number)
      print(info)


  while True:
      user_angle = p.readUserDebugParameter(angle)
      user_throttle = p.readUserDebugParameter(throttle)
      for joint_index in wheel_indices:
          p.setJointMotorControl2(car, joint_index,
                                  p.VELOCITY_CONTROL,
                                  targetVelocity=user_throttle)
      for joint_index in hinge_indices:
          p.setJointMotorControl2(car, joint_index,
                                  p.POSITION_CONTROL, 
                                  targetPosition=user_angle)
      p.stepSimulation()

  p.disconnect()

main()