import pybullet as pb
import time
import pybullet_data
import random, numpy, math, time
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display, HTML
import pybullet as pb
# import tensorflow as tf
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
# from tensorflow.python.keras.layers import LSTM
# from tensorflow.python.keras.optimizers import RMSprop

####################
# Окружение (игра)
####################

MAX_STEPS = 1000  # максимальное количество шагов симуляции
STEPS_AFTER_TARGET = 30  # количество шагов симуляции после достижения цели
TARGET_DELTA = 0.2  # величина приемлемого качения возле цели (абсолютное значение)
FORCE_DELTA = 0.1  # шаг измениния силы (абсолютное значение)
PB_BallMass = 1  # масса шара
PB_BallRadius = 0.2  # радиус шара
PB_HEIGHT = 10  # максимальная высота поднятия шара
MAX_FORCE = 20  # максимальная вертикальная сила пиложенная к шару
MIN_FORCE = 0  # минимальнпая сила пиложенная к шару
MAX_VEL = 14.2  # максимальная вертикальная скорость шара
MIN_VEL = -14.2  # минимальная вертикальная скорость шара


class Environment:
    def __init__(self):
        # текущее состояние окружения
        self.pb_z = 0  # текущая высота шара
        self.pb_force = 0  # текущая сила приложенная к шару
        self.pb_velocity = 0  # текущая вертикальная скорость шара
        self.z_target = 0  # целевая высота
        self.start_time = 0  # время начала новой игры
        self.steps = 0  # количество шагов после начала симуляции
        self.target_area = 0  # факт достежения цели
        self.steps_after_target = 0  # количество шагов после достежения цели

        # создадим симуляцию
        self.pb_physicsClient = pb.connect(pb.GUI)

    def reset(self):
        # случайные высота шара и целевая высота
        z_target = random.uniform(0.01, 0.99)
        self.z_target = PB_BallRadius + z_target * PB_HEIGHT
        z = random.uniform(0.05, 0.95)
        self.pb_z = PB_BallRadius + z * PB_HEIGHT

        # сброс параметров окружения
        pb.resetSimulation()
        self.target_area = 0
        self.start_time = time.time()
        self.steps = 0
        self.steps_after_target = 0

        # шаг симуляции 1/60 сек.
        pb.setTimeStep(1. / 60)

        # поверхность
        floorColShape = pb.createCollisionShape(pb.GEOM_PLANE)
        # для GEOM_PLANE, visualShape - не отображается, будем использовать GEOM_BOX
        floorVisualShapeId = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[100, 100, 0.0001], rgbaColor=[1, 1, .98, 1])
        self.pb_floorId = pb.createMultiBody(0, floorColShape, floorVisualShapeId, [0, 0, 0],
                                             [0, 0, 0, 1])  # (mass,collisionShape,visualShape)

        # шар
        ballPosition = [0, 0, self.pb_z]
        ballOrientation = [0, 0, 0, 1]
        ballColShape = pb.createCollisionShape(pb.GEOM_SPHERE, radius=PB_BallRadius)
        ballVisualShapeId = pb.createVisualShape(pb.GEOM_SPHERE, radius=PB_BallRadius, rgbaColor=[0.25, 0.75, 0.25, 1])
        self.pb_ballId = pb.createMultiBody(PB_BallMass, ballColShape, ballVisualShapeId, ballPosition,
                                            ballOrientation)  # (mass, collisionShape, visualShape, ballPosition, ballOrientation)
        # pb.changeVisualShape(self.pb_ballId,-1,rgbaColor=[1,0.27,0,1])

        # указатель цели (без CollisionShape, только отображение(VisualShape))
        targetPosition = [0, 0, self.z_target]
        targetOrientation = [0, 0, 0, 1]
        targetVisualShapeId = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[1, 0.025, 0.025], rgbaColor=[0, 0, 0, 1])
        self.pb_targetId = pb.createMultiBody(0, -1, targetVisualShapeId, targetPosition, targetOrientation)

        # гравитация
        pb.setGravity(0, 0, -10)

        # ограничим движение шара только по вертикальной оси
        pb.createConstraint(self.pb_floorId, -1, self.pb_ballId, -1, pb.JOINT_PRISMATIC, [0, 0, 1], [0, 0, 0],
                            [0, 0, 0])

        # установим действующую силу на шар, чтобы компенсировать гравитацию
        self.pb_force = 10 * PB_BallMass
        pb.applyExternalForce(self.pb_ballId, -1, [0, 0, self.pb_force], [0, 0, 0], pb.LINK_FRAME)

        # return values
        observation = self.getObservation()
        reward, done = self.getReward()
        info = self.getInfo()
        return [observation, reward, done, info]

    # Наблюдения (возвращаются нормализованными)
    def getObservation(self):
        # расстояние до цели
        d_target = 0.5 + (self.pb_z - self.z_target) / (2 * PB_HEIGHT)
        # действующая сила
        force = (self.pb_force - MIN_FORCE) / (MAX_FORCE - MIN_FORCE)
        # текущая высота шара
        z = (self.pb_z - PB_BallRadius) / PB_HEIGHT
        # текущая скорость
        z_velocity = (self.pb_velocity - MIN_VEL) / (MAX_VEL - MIN_VEL)
        state = [d_target, force, z_velocity]
        return state

    # вычисление награды за действие
    def getReward(self):
        done = False
        z_reward = 0
        # Факт достижения цели, после чего ждем STEPS_AFTER_TARGET шагов и завершем игру.
        if (TARGET_DELTA >= math.fabs(self.z_target - self.pb_z)):
            self.target_area = 1
            z_reward = 1
        # Выход за пределы зоны
        if (self.pb_z > (PB_HEIGHT + PB_BallRadius) or self.pb_z < PB_BallRadius):
            done = True
        # Завершение игры после достижения цели
        if (self.target_area > 0):
            self.steps_after_target += 1
            if (self.steps_after_target >= STEPS_AFTER_TARGET):
                done = True
        # Завершение игры по таймауту
        if (self.steps >= MAX_STEPS):
            done = True

        return [z_reward, done]

    # Дополнительная информация для сбора статистики
    def getInfo(self):
        game_time = time.time() - self.start_time
        if game_time:
            fps = round(self.steps / game_time)
        return {'step': self.steps, 'fps': fps}

    # Запуск шага симуляции согласно переданному действию
    def step(self, action):
        self.steps += 1
        if action == 0:
            # 0 - увеличение приложеной силы
            self.pb_force -= FORCE_DELTA
            if self.pb_force < MIN_FORCE:
                self.pb_force = MIN_FORCE
        else:
            # 1 - уменьшение приложенной силы
            self.pb_force += FORCE_DELTA
            if self.pb_force > MAX_FORCE:
                self.pb_force = MAX_FORCE

        # изменим текущую сил и запустим шаг симуляции
        pb.applyExternalForce(self.pb_ballId, -1, [0, 0, self.pb_force], [0, 0, 0], pb.LINK_FRAME)
        pb.stepSimulation()

        # обновим парамтры состояния окружения (положение и скорость шара)
        curPos, curOrient = pb.getBasePositionAndOrientation(self.pb_ballId)
        lin_vel, ang_vel = pb.getBaseVelocity(self.pb_ballId)
        self.pb_z = curPos[2]
        self.pb_velocity = lin_vel[2]

        # вернем наблюдения, награду, факт окончания игры и доп.информацию
        observation = self.getObservation()
        reward, done = self.getReward()
        info = self.getInfo()
        return [observation, reward, done, info]

    # Текущее изображение с камеры
    def render(self):
        camTargetPos = [0, 0, 5]  # расположение цели (фокуса) камеры
        camDistance = 10  # дистанция камеры от цели
        yaw = 0  # угол рыскания относительно цели
        pitch = 0  # наклон камеры относительно цели
        roll = 0  # угол крена камеры относительно цели
        upAxisIndex = 2  # ось вертикали камеры (z)

        fov = 60  # угол зрения камеры
        nearPlane = 0.01  # расстояние до ближней плоскости отсечения
        farPlane = 20  # расстояние до дальной плоскости отсечения
        pixelWidth = 320  # ширина изображения
        pixelHeight = 200  # высота изображения
        aspect = pixelWidth / pixelHeight  # соотношение сторон изображения

        # видовая матрица
        viewMatrix = pb.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch, roll, upAxisIndex)
        # проекционная матрица
        projectionMatrix = pb.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane);
        # рендеринг изображения с камеры
        img_arr = pb.getCameraImage(pixelWidth, pixelHeight, viewMatrix, projectionMatrix, shadow=0,
                                    lightDirection=[0, 1, 1], renderer=pb.ER_TINY_RENDERER)
        w = img_arr[0]  # width of the image, in pixels
        h = img_arr[1]  # height of the image, in pixels
        rgb = img_arr[2]  # color data RGB
        dep = img_arr[3]  # depth data

        # вернем rgb матрицу
        return rgb

env = Environment()
env.reset()
# env.render()
for i in range (10000):
    pb.stepSimulation()
    time.sleep(1./60.)
# cubePos, cubeOrn = pb.getBasePositionAndOrientation(boxId)
# print(cubePos,cubeOrn)
pb.disconnect()