import numpy as np


class DifferentialDriveRobotWithoutNoise:

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

        print("Robot is initial without noise")

    def update(self, v, omega, dt):
        # 更新朝向
        self.theta += omega * dt

        # 规范化角度到[-π, π]
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))

        # 更新位置
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt

        # 记录轨迹
        self.trajectory.append((self.x, self.y, self.theta))

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
