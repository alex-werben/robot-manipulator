from robot import Robot
from src.dynamixel import Dynamixel
import numpy as np
import time
import mujoco.viewer
import mujoco
from simulation.interface import SimulatedRobot
import threading
# from mujoco import mjvi
# def read_follower_position():
#     global target_pos
#     while True:
#         target_pos = np.array(follower.read_position())
#         target_pos = (target_pos / 2048 - 1) * 3.14
#         target_pos[1] = -target_pos[1]
#         target_pos[3] = -target_pos[3]
#         target_pos[4] = -target_pos[4]

follower_dynamixel = Dynamixel.Config(baudrate=1_000_000, device_name="/dev/tty.usbmodem58FA0959341").instantiate()
follower = Robot(follower_dynamixel, servo_ids=[1, 2, 3, 4, 5, 6])
follower.tmp_disable_torque()
# follower._enable_torque()

# while True:
#     print(follower.read_position())
#     time.sleep(0.01)



path = "urdf/scene.xml"

m = mujoco.MjModel.from_xml_path(path)
d = mujoco.MjData(m)

r = SimulatedRobot(m, d)

# with mujoco.viewer.launch_passive(m, d) as viewer:
#     start = time.time()
#     while viewer.is_running():

#         step_start = time.time()

#         sim_robot_position = r.read_position()
#         print(r._pos2pwm(sim_robot_position))
#         joint_position = r._pos2pwm(sim_robot_position)
#         follower.set_goal_pos(joint_position)

#         mujoco.mj_step(m, d)
#         viewer.sync()

#         time_until_next_step = m.opt.timestep - (time.time() - step_start)
#         if time_until_next_step > 0:
#             time.sleep(time_until_next_step)
