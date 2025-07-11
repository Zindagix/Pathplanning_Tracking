import numpy as np
import random


# 卡尔曼滤波器类
class KalmanFilter:

    def __init__(self, dt=0.1, state_variance=1.0, measurement_variance=1.0):
        self.dt = dt  # 时间步长
        self.A = np.array([[1, self.dt], [0, 1]])  # 状态转移矩阵
        self.H = np.array([[1, 0]])  # 测量矩阵
        self.Q = state_variance * np.eye(2)  # 过程噪声协方差矩阵
        self.R = measurement_variance  # 测量噪声协方差
        self.x = np.zeros((2, 1))  # 初始状态估计
        self.P = np.eye(2)  # 初始估计协方差
        self.alpha = 1  # 自适应系数

    def predict(self):
        self.x = np.dot(self.A, self.x)  # 预测状态
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q  # 预测协方差

    def update(self, z):
        y = z - np.dot(self.H, self.x)  # 计算残差
        S1 = np.dot(self.H, np.dot(self.P, self.H.T))
        S = S1 + self.R  # 计算残差协方差
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # 计算卡尔曼增益
        self.x = self.x + np.dot(K, y)  # 更新状态估计
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)  # 更新协方差
        y1 = z - np.dot(self.H, self.x)
        Ky = np.dot(K, y)
        self.R = self.alpha * self.R + (1 - self.alpha) * (np.dot(y1, y1.T) +
                                                           S1)
        self.Q = self.alpha * self.Q + (1 - self.alpha) * np.dot(Ky, Ky.T)

    def get_state(self):
        return self.x.flatten()  # 返回当前状态


class DifferentialDriveRobot:

    def __init__(self, x=None, y=None, theta=0, wheel_base=20.0):
        # 初始位置和朝向
        self.x = x
        self.y = y
        self.theta = theta  # 朝向（弧度）

        # 机器人参数
        self.wheel_base = wheel_base  # 轮距
        self.wheel_radius = 5.0  # 轮子半径

        # 状态历史记录
        if self.x and self.y:
            self.trajectory = [(x, y, theta)]
        else:
            self.trajectory = []

        # 初始化KalmanFilter
        self.x_kf = KalmanFilter(dt=0.1,
                                 state_variance=1.0,
                                 measurement_variance=1.0)
        self.y_kf = KalmanFilter(dt=0.1,
                                 state_variance=1.0,
                                 measurement_variance=1.0)
        self.theta_kf = KalmanFilter(dt=0.1,
                                     state_variance=100,
                                     measurement_variance=1.0)

        self.x_kf.x = np.array([[x], [0]])
        self.y_kf.x = np.array([[y], [0]])
        self.theta_kf.x = np.array([[theta], [0]])

        # 设置robot移动噪声
        self.xy_noise = 0.5
        self.theta_noise = 0.1

        # 设置robot轮子角度偏移
        self.omega_offset = 0.4

        print("Robot is initial with noise")

    def set_noise(self, xy_noise, theta_noise):
        self.xy_noise = xy_noise
        self.theta_noise = theta_noise

    def set_omega_offset(self, omega_offset):
        self.omega_offset = omega_offset

    def update(self, v, omega, dt):

        # 添加高斯分布噪声
        v = random.gauss(v, self.xy_noise)
        omega = random.gauss(omega, self.theta_noise)

        # 轮子转角添加系统误差
        omega += self.omega_offset

        # 更新朝向
        self.theta += omega * dt

        # 规范化角度到[-pi, pi]
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))

        # 更新位置
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt

        # 记录轨迹
        self.trajectory.append((self.x, self.y, self.theta))

        # 更新KalmanFilter
        self.x_kf.predict()
        self.y_kf.predict()
        self.theta_kf.predict()

        self.x_kf.update(self.x)
        self.y_kf.update(self.y)
        self.theta_kf.update(self.theta)

        # 获取KalmanFilter当前状态
        self.x, _ = self.x_kf.get_state()
        self.y, _ = self.y_kf.get_state()
        self.theta, _ = self.theta_kf.get_state()

    def get_wheel_speeds(self, v, omega):
        # 计算左右轮速度
        left_speed = v - omega * self.wheel_base / 2
        right_speed = v + omega * self.wheel_base / 2
        return left_speed, right_speed

    def get_position(self):
        return (self.x, self.y, self.theta)

    def reset_position(self, x, y, theta=0):
        # 重置状态
        self.x = x
        self.y = y
        self.theta = theta
        self.trajectory = [(self.x, self.y, self.theta)]

        self.x_kf.x = np.array([[x], [0]])
        self.y_kf.x = np.array([[y], [0]])
        self.theta_kf.x = np.array([[theta], [0]])
