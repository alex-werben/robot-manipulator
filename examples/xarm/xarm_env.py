import numpy as np
import pybullet as p
import pybullet_data as pd
import time
import math


class Env:
	def __init__(self, rts=1):

		p.connect(p.GUI)  # , options="--background_color_red=1.0 --background_color_blue=1.0 --background_color_green=1.0")
		p.setAdditionalSearchPath(pd.getDataPath())
		print(pd.getDataPath())
		self.REAL_TIME_SIMULATION_FLAG = rts
		useFixedBase = True
		flags = 0

		plane_pos = [0,0,0]
		plane = p.loadURDF("plane.urdf", plane_pos, flags = flags, useFixedBase=useFixedBase)
		table_pos = [0, 0, -0.625]
		self.tableId = p.loadURDF("table/table.urdf", table_pos, flags=flags, useFixedBase=useFixedBase)
		self.xarmId = p.loadURDF("xarm/xarm6_robot.urdf", flags=flags, useFixedBase=useFixedBase)
		# self.xarmId = p.loadURDF("xarm/xarm6_with_gripper.urdf", [0, 0, 0])


	def get_info(self):
		self.jointIds = []
		self.paramIds = []
		rp = [0, 0.5 * math.pi, 0, 0, 0, 0, 0]
		for i in range(p.getNumJoints(self.xarmId)):
			p.resetJointState(self.xarmId, i, rp[i])
		self.gripperId = 6
		self.lastJointId = p.getNumJoints(self.xarmId) - 1
		self.movableJoints = 0

		self.gripperJointIds = []

		for j in range(p.getNumJoints(self.xarmId)):
			# p.changeDynamics(self.xarmId, j, linearDamping=0, angularDamping=0)
			info = p.getJointInfo(self.xarmId, j)

			jointName: str = info[1]
			jointType = info[2]

			if jointName.startswith("joint") and (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
				self.jointIds.append(j)
				self.paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), -4, 4, 0))
				self.movableJoints += 1

			if jointName.endswith("joint") and (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
				self.gripperJointIds.append(j)



	# print(p.getLinkState(self.xarmId, self.gripperId))

	def calculate_joint_poses(self, position):

		orn = p.getQuaternionFromEuler([0, -math.pi, 0])
		jointPoses = p.calculateInverseKinematics(self.xarmId,
													self.lastJointId,
													position,
													orn)
		return jointPoses

	def simulate_debug(self):
		while True:
			p.stepSimulation()
			for i in range(len(self.paramIds)):
				c = self.paramIds[i]
				targetPos = p.readUserDebugParameter(c)
				p.setJointMotorControl2(self.xarmId, self.jointIds[i], p.POSITION_CONTROL, targetPos, force=5 * 240.)
			time.sleep(1. / 10000.)

	def move_to_position(self, endPos):
		currentPos = np.array(p.getLinkState(self.xarmId, self.gripperId)[0])
		endPos = np.array(endPos)
		ax_id = 0
		ax = ['x', 'y', 'z']
		eps = 1e-2
		cnt = 0
		while not np.all(np.abs(endPos - currentPos) < eps):
			while not np.abs(endPos[ax_id] - currentPos[ax_id]) < eps:
				currentPos[ax_id] += eps if endPos[ax_id] > currentPos[ax_id] else -eps
				cnt += 1

				if cnt % 100 == 0:
					print(currentPos)
				jointPoses = self.calculate_joint_poses(currentPos)
				for j in range(self.movableJoints):
					p.setJointMotorControl2(bodyIndex=self.xarmId,
											jointIndex=j + 1,
											controlMode=p.POSITION_CONTROL,
											targetPosition=jointPoses[j],
											# targetVelocity=0,
											force=5000,
											# positionGain=0.03,
											# velocityGain=1
											)

				linkPose = p.getLinkState(self.xarmId, self.gripperId)[0]
				p.addUserDebugLine(currentPos, linkPose, [0.7, 0, 0], 1, self.trailDuration)
				currentPos[ax_id] = linkPose[ax_id]
				time.sleep(1. / 500.)
				p.stepSimulation()
			print(f"{ax[ax_id]} reached, current_pos: {currentPos}, end_pos: {endPos}")
			ax_id += 1


		print(f"End position reached: {endPos}")

	def simulate(self):
		p.setRealTimeSimulation(0)
		self.trailDuration = 50

		jointPoses = p.calculateInverseKinematics(self.xarmId,
													self.gripperJointIds[-1],
													[0.3, 0.3, 0.3],
													p.getQuaternionFromEuler([0, -math.pi, 0]))
		print(jointPoses)
		# pos = [0, 0.1, 0.05]
		# prevPose = pos.copy()
		# positions = [
		# 	# [0, 0.2, 0.05],
		# 	[0, 0.3, 0.1],
		# 	[0.3, 0.3, 0.1],
		# 	[0.3, 0.3, 0.3],
		# 	[-0.3, 0.3, 0.3],
		# 	[-0.3, -0.3, 0.3],
		# 	[0.3, -0.3, 0.3],
		# 	[0.3, 0.3, 0.3],
		# ]
		# for pos in positions:
		# 	self.move_to_position(pos)
		#
		# p.disconnect()


if __name__ == "__main__":
	env = Env()
	env.get_info()
	# env.calculate_joint_poses()
	env.simulate_debug()

