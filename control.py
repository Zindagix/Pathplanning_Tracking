import numpy as np
import pygame
from scipy.optimize import minimize


class PIDController:

    def __init__(self, kp=0.4, ki=0.01, kd=0.1):
        self.kp = kp  # 比例系数
        self.ki = ki  # 积分系数
        self.kd = kd  # 微分系数

        self.prev_error_distance = 0
        self.integral_distance = 0
        self.prev_error_omega = 0
        self.integral_omega = 0

    def compute_control(self, x, y, theta, target_x, target_y):
        # 计算位置误差
        dx = target_x - x
        dy = target_y - y
        error_distance = np.sqrt(dx**2 + dy**2)

        # 计算期望角度
        desired_theta = np.arctan2(dy, dx)

        # 计算角度误差（考虑角度环绕）
        error_omega = np.arctan2(np.sin(desired_theta - theta),
                                 np.cos(desired_theta - theta))

        # PID计算
        self.integral_distance += error_distance
        derivative_error_distance = error_distance - self.prev_error_distance
        self.prev_error_distance = error_distance

        self.integral_omega += error_omega
        derivative_error_omega = error_omega - self.prev_error_omega
        self.prev_error_omega = error_omega

        # 线速度控制
        v = self.kp * error_distance + self.ki * self.integral_distance + self.kd * derivative_error_distance
        v = np.clip(v, 0, 5)  # 最大速度限制

        # 角速度控制
        omega = self.kp * error_omega + self.ki * self.integral_omega + self.kd * derivative_error_omega

        # 当角度误差大时减速
        if abs(error_omega) > np.pi / 4:
            v *= 0.5

        return v, omega


class MPCController:

    def __init__(self, N=7, dt=0.01, max_speed=26.0, max_omega=2.0):
        self.N = N  # 预测时域
        self.dt = dt  # 时间步长
        self.max_speed = max_speed
        self.max_omega = max_omega

    def robot_model(self, state, u):
        # 状态: [x, y, theta]
        # 控制: [v, omega]
        x, y, theta = state
        v, omega = u

        # 差分驱动模型
        x_dot = v * np.cos(theta)
        y_dot = v * np.sin(theta)
        theta_dot = omega

        return np.array([x_dot, y_dot, theta_dot])

    def predict_trajectory(self, init_state, u_sequence):
        state = init_state.copy()
        trajectory = [state]

        for i in range(self.N):
            u = u_sequence[i]
            state_dot = self.robot_model(state, u)

            state = state + state_dot * self.dt
            trajectory.append(state)

        return np.array(trajectory)

    def cost_function(self, u_sequence, init_state, ref_trajectory):
        # 将控制序列转换为向量
        u_sequence = u_sequence.reshape((self.N, 2))

        # 预测轨迹
        trajectory = self.predict_trajectory(init_state, u_sequence)

        # 轨迹跟踪误差
        tracking_error = 0
        for i in range(1, self.N + 1):
            # 参考轨迹上的点（线性插值）
            ref_idx = min(i, len(ref_trajectory) - 1)
            ref_point = ref_trajectory[ref_idx]

            # 位置误差
            pos_error = np.linalg.norm(trajectory[i][:2] - ref_point[:2])
            tracking_error += pos_error

            # 角度误差
            angle_error = abs(
                np.arctan2(np.sin(trajectory[i][2] - ref_point[2]),
                           np.cos(trajectory[i][2] - ref_point[2])))
            tracking_error += 0.1 * angle_error

        # 控制量惩罚
        control_penalty = 0
        for i in range(self.N):
            # 控制量变化惩罚
            if i > 0:
                v_diff = u_sequence[i, 0] - u_sequence[i - 1, 0]
                omega_diff = u_sequence[i, 1] - u_sequence[i - 1, 1]
                control_penalty += 0.1 * (v_diff**2 + omega_diff**2)

            # 控制量大小惩罚
            control_penalty += 0.0005 * (u_sequence[i, 0]**2 +
                                         u_sequence[i, 1]**2)

        return tracking_error + control_penalty

    def compute_control(self, x, y, theta, target_x, target_y):
        # 生成参考轨迹（直线到目标点）
        ref_trajectory = []
        for i in range(self.N + 1):
            t = i / self.N
            ref_x = x + t * (target_x - x)
            ref_y = y + t * (target_y - y)
            ref_theta = np.arctan2(target_y - y, target_x - x)
            ref_trajectory.append([ref_x, ref_y, ref_theta])

        # 初始状态
        init_state = np.array([x, y, theta])

        # 初始控制序列（零控制）
        u0 = np.zeros(self.N * 2)

        # 约束条件
        bounds = []
        for i in range(self.N):
            bounds.append((0, self.max_speed))  # v约束
            bounds.append((-self.max_omega, self.max_omega))  # omega约束

        # 优化求解
        res = minimize(self.cost_function,
                       u0,
                       args=(init_state, ref_trajectory),
                       bounds=bounds,
                       method='SLSQP',
                       options={
                           'maxiter': 100,
                           'ftol': 1e-6
                       })

        # 获取最优控制序列
        u_opt = res.x

        # 返回第一个控制量
        v = u_opt[0]
        omega = u_opt[1]

        # 当角度误差大时减速
        if abs(omega) > np.pi / 4:
            v *= 0.5
        return v, omega


class TRACKER:

    def __init__(self, win, grid, rows, width, isgrid, route, robot, vis):
        self.robot = robot
        self.tracking_errors = {}
        self.vis = vis
        self.win = win
        self.grid = grid
        self.rows = rows
        self.width = width
        self.isgrid = isgrid
        self.route = route
        self.interpolate_route = []

    def vectorized_distance(self, p0, p1, p2):
        v = p2 - p1
        w = p0 - p1
        c1 = np.dot(w, v)
        if c1 <= 0:
            return np.linalg.norm(p0 - p1)
        c2 = np.dot(v, v)
        if c2 <= c1:
            return np.linalg.norm(p0 - p2)
        return np.linalg.norm(p0 - (p1 + (c1 / c2) * v))

    # 计算与规划路径的误差
    def calcu_errors2paths(self):
        if len(self.route) < 2:
            if len(self.route) == 1:
                return np.sqrt((self.robot.x - self.route[0].x -
                                self.route[0].width // 2)**2 +
                               (self.robot.y - self.route[0].y -
                                self.route[0].width // 2)**2)
            else:
                return float("inf")
        min_error = float("inf")
        for i in range(len(self.route) - 1):
            min_error = min(
                min_error,
                self.vectorized_distance(
                    np.array((self.robot.x, self.robot.y)),
                    np.array((self.route[i].x + self.route[i].width // 2,
                              self.route[i].y + self.route[i].width // 2)),
                    np.array(
                        (self.route[i + 1].x + self.route[i + 1].width // 2,
                         self.route[i + 1].y + self.route[i + 1].width // 2))))
        return min_error

    # 路径插值
    def interpolate_points(self, points, num=9):
        if len(points) < 1:
            return []

        interpolated = []

        for i in range(len(points) - 1):
            x0, y0 = points[i].x + points[i].width // 2, points[
                i].y + points[i].width // 2
            x1, y1 = points[i + 1].x + points[i + 1].width // 2, points[
                i + 1].y + points[i + 1].width // 2

            interpolated.append((x0, y0))

            for j in range(1, num + 1):
                r = j / (num + 1)
                x = x0 + r * (x1 - x0)
                y = y0 + r * (y1 - y0)
                interpolated.append((x, y))

        interpolated.append((points[-1].x + points[-1].width // 2,
                             points[-1].y + points[-1].width // 2))
        return interpolated

    # 获取鼠标点击位置
    def get_clicked_pos(self, pos):
        gap = self.width // self.rows
        y, x = pos

        row = y // gap
        col = x // gap

        return row, col

    def run_tracking(self, control_method, is_reset=True, can_operate=True):

        if not self.route:
            print("路径未规划")
            print()
            return [], []

        # self.interpolate_route = self.interpolate_points(self.route)
        # 重置机器人位置
        if is_reset:
            self.robot.reset_position(
                self.route[-1].x + self.route[-1].width // 2,
                self.route[-1].y + self.route[-1].width // 2, 0)

        # 创建控制器
        if control_method == "PID":
            controller = PIDController(kp=0.5, ki=0.01, kd=0.1)
            dt = 0.1
        else:  # MPC
            controller = MPCController()
            dt = 0.1

        errors = []
        errors2paths = []
        current_routepoint = len(self.route) - 1

        self.vis.count = 0
        count = 0

        self.vis.draw(self.win,
                      self.grid,
                      self.rows,
                      self.width,
                      self.isgrid,
                      self.route,
                      robot=self.robot)
        # 模拟跟踪过程
        while current_routepoint >= 0:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("manual quit")
                    exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    print("manual escape")
                    return errors, errors2paths

            if can_operate and not self.route[current_routepoint].is_end():
                # 事件处理
                mouse_pressed = pygame.mouse.get_pressed()
                if mouse_pressed[0] or mouse_pressed[2]:
                    pos = pygame.mouse.get_pos()
                    row, col = self.get_clicked_pos(pos)
                    if 0 <= row < self.rows and 0 <= col < self.rows:  # 添加边界检查
                        node = self.grid[row][col]
                        if not (node.x <= self.robot.x <= node.x + node.width
                                and
                                node.y <= self.robot.y <= node.y + node.width
                                ) and self.route[current_routepoint] != node:
                            if mouse_pressed[0]:  # 左键
                                if not node.is_end() and not node.is_start(
                                ) and not node.is_black_barrier():
                                    node.make_barrier()
                                    return -1, node, self.route[
                                        current_routepoint]
                            else:  # 右键
                                if node.is_barrier():
                                    node.reset(isstrict=True)
                                    return -2, node, self.route[
                                        current_routepoint]
            # 获取当前目标点
            target_x = self.route[current_routepoint].x + self.route[
                current_routepoint].width // 2
            target_y = self.route[current_routepoint].y + self.route[
                current_routepoint].width // 2

            # 计算控制命令
            v, omega = controller.compute_control(self.robot.x, self.robot.y,
                                                  self.robot.theta, target_x,
                                                  target_y)

            # 更新机器人状态
            self.robot.update(v, omega, dt)

            # 计算跟踪误差
            error = np.sqrt((self.robot.x - target_x)**2 +
                            (self.robot.y - target_y)**2)
            errors.append(error)
            errors2paths.append(self.calcu_errors2paths())

            # 检查是否到达目标点
            if current_routepoint and error < self.route[
                    current_routepoint].width / 4:
                current_routepoint -= 1
            elif current_routepoint == 0 and error < 2 and control_method == 'PID':
                current_routepoint -= 1
            elif current_routepoint == 0 and error < 2 and control_method == 'MPC':
                current_routepoint -= 1

            # 渲染画面
            if count % 4 == 0:
                if count % 16 == 0 or control_method != 'PID':

                    self.vis.draw_tracking(self.win, self.grid, self.rows,
                                           self.width, self.isgrid, self.route,
                                           self.robot, target_x, target_y)

                    # 显示信息
                    self.vis.draw_errortext(self.win, 10, 50, control_method,
                                            error)
                    self.vis.draw_dynamic_errors(self.win, errors,
                                                 errors2paths, control_method)

            count += 1
        self.tracking_errors[control_method] = errors

        print("跟踪结束")
        print()
        return errors, errors2paths
