import argparse
import pygame
import json
import random
import numpy as np
import matplotlib.pyplot as plt

from visualization import Visualization, add_bottom_labels
from path_planning import Astar_algorithm, Dijkstra_algorithm, Dstar_lite, BFS
from robot_model_with_noise import DifferentialDriveRobot
from robot_model_without_noise import DifferentialDriveRobotWithoutNoise
from control import TRACKER

fig = plt.figure(figsize=(13, 8))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

parser = argparse.ArgumentParser()
parser.add_argument(
    '-f',
    '--file',
    type=str,
    default="grid_save_incline.json",
    help="the json file of the map, default is 'grid_save_incline.json'")
parser.add_argument('--noise',
                    action="store_true",
                    help='whether to set noise to the robot or not')
args = parser.parse_args()

# 初始化
pygame.init()
WIDTH, HEIGHT = 1000, 1000
OFFSET = 700
ROWS = 50
WIN = pygame.display.set_mode((WIDTH + OFFSET, HEIGHT))
pygame.display.set_caption("移动机器人路径规划与跟踪系统")

# 创建可视化对象
VIS = Visualization(WIDTH, OFFSET)


class Application:

    def __init__(self):
        self.clock = pygame.time.Clock()
        self.width = WIDTH
        self.rows = ROWS
        self.grid = VIS.make_grid(self.rows, self.width)
        self.is_show_grid = False

        self.start = None
        self.end = None

        self.isrun = True
        self.is_algorithm_running = False

        # 初始路径规划算法和控制算法
        self.algorithm = "D*lite"
        self.controller_type = "PID"

        # 存路径
        self.route = []

        # 存变化的节点，主要用于dstar_lite算法
        self.changed_nodes = []
        self.removed_nodes = []

        self.robot = DifferentialDriveRobot(
        ) if args.noise else DifferentialDriveRobotWithoutNoise()
        self.dstarlite = None

        self.planning_iters = {}
        self.planning_times = {}
        self.path_lengths = {}
        self.path_length = 0
        self.tracking_errors = {}
        self.algorithms = ["D*lite", "D*lite_replan", "A*", "Dijkstra", "BFS"]

    # 四个路径规划算法
    def running_algorithm(self):
        if not (self.start and self.end and not self.is_algorithm_running):
            if not self.start: print("起点未设置")
            if not self.end: print("终点未设置")
            print()
            return
        self.dstarlite = None
        self.route = []
        self.changed_nodes = []
        self.removed_nodes = []

        self.robot.reset_position(self.start.x + self.start.width // 2,
                                  self.start.y + self.start.width // 2, 0)
        if self.algorithm == 'A*':
            self.is_algorithm_running = True
            VIS.reset_grid(self.grid)
            for row in self.grid:
                for node in row:
                    node.update_neighbors(self.grid)

            init_iter, init_time, route = Astar_algorithm(
                lambda: VIS.draw(WIN,
                                 self.grid,
                                 self.rows,
                                 self.width,
                                 self.is_show_grid,
                                 robot=self.robot), self.start, self.end)
            if route == False:
                print("there is no accessible route.")
                route = []
            self.is_algorithm_running = False
            self.route = route

        elif self.algorithm == 'D*lite':
            self.is_algorithm_running = True
            VIS.reset_grid(self.grid)
            for row in self.grid:
                for node in row:
                    node.update_neighbors(self.grid)

            # 创建D*实例
            self.dstarlite = Dstar_lite(
                self.grid, self.start, self.end,
                lambda: VIS.draw(WIN,
                                 self.grid,
                                 self.rows,
                                 self.width,
                                 self.is_show_grid,
                                 robot=self.robot))

            # 初始规划
            init_iter, init_time, success = self.dstarlite.init_plan()
            print(
                f"D*lite初始规划: 迭代次数={init_iter}, 时间={init_time:.4f}s, 是否找到路径={bool(success)}"
            )
            # 重建并可视化初始路径
            if success:
                route = self.dstarlite.reconstruct_path()
            elif success == False:
                print("there is no accessible route.")
                route = []
                self.dstarlite = None
            else:
                route = []
                self.dstarlite = None
            self.is_algorithm_running = False
            self.route = route

        elif self.algorithm == 'Dijkstra':
            self.is_algorithm_running = True
            VIS.reset_grid(self.grid)
            for row in self.grid:
                for node in row:
                    node.update_neighbors(self.grid)

            init_iter, init_time, route = Dijkstra_algorithm(
                lambda: VIS.draw(WIN,
                                 self.grid,
                                 self.rows,
                                 self.width,
                                 self.is_show_grid,
                                 robot=self.robot), self.start, self.end)
            if route == False:
                print("there is no accessible route.")
                route = []
            self.is_algorithm_running = False
            self.route = route
        elif self.algorithm == 'BFS':
            self.is_algorithm_running = True
            VIS.reset_grid(self.grid)
            for row in self.grid:
                for node in row:
                    node.update_neighbors(self.grid)

            init_iter, init_time, route = BFS(
                lambda: VIS.draw(WIN,
                                 self.grid,
                                 self.rows,
                                 self.width,
                                 self.is_show_grid,
                                 robot=self.robot), self.start, self.end)
            if route == False:
                print("there is no accessible route.")
                route = []
            self.is_algorithm_running = False
            self.route = route

        print()
        self.planning_iters[self.algorithm] = init_iter
        self.planning_times[self.algorithm] = init_time
        self.path_length = self.start.g_score if self.algorithm == "D*lite" else self.end.g_score

        self.path_lengths[self.algorithm] = self.path_length

    # dstar_lite重规划
    def dstar_lite_replan(self, is_reset=True):

        if not self.dstarlite:
            print("需(再次)(完整)运行D*lite算法")
            print()
            return

        if self.dstarlite:

            if self.dstarlite.end != self.end:
                return
            if not (self.changed_nodes or self.removed_nodes
                    or self.dstarlite.start != self.start):
                print("地图无变化")
            if self.dstarlite.start != self.start:
                self.dstarlite.start = self.start

            for node in self.route:
                if not node.is_barrier() and not node.is_end():
                    node.make_closed()
            self.route = []
            if not (self.start and self.end and not self.is_algorithm_running):
                return
            if is_reset:
                self.robot.reset_position(self.start.x + self.start.width // 2,
                                          self.start.y + self.start.width // 2,
                                          0)
            self.is_algorithm_running = True

            for row in self.grid:
                for node in row:
                    node.update_neighbors(self.grid)

            self.dstarlite.add_obstacle(self.changed_nodes)
            self.dstarlite.remove_obstacle(self.removed_nodes)

            # 重规划
            replan_iter, replan_time, success = self.dstarlite.replan()
            print(
                f"D*lite重规划: 迭代次数={replan_iter}, 时间={replan_time:.4f}s, 是否找到路径={bool(success)}"
            )

            # 重建并可视化初始路径
            if success:
                route = self.dstarlite.reconstruct_path()
            elif success == False:
                print("there is no accessible route.")
                route = []
            else:
                route = []
            self.is_algorithm_running = False
            self.route = route
            self.changed_nodes = []
            self.removed_nodes = []

            print()
            self.planning_iters["D*lite_replan"] = replan_iter
            self.planning_times["D*lite_replan"] = replan_time
            self.path_length = self.start.g_score
            self.path_lengths["D*lite_replan"] = self.start.g_score

    # 获取鼠标点击位置
    def get_clicked_pos(self, pos):
        gap = self.width // self.rows
        y, x = pos

        row = y // gap
        col = x // gap

        return row, col

    def save_grid(self):
        grid_data = {"rows": self.rows, "start": None, "end": None, "grid": []}

        # 记录起点终点位置
        if self.start:
            grid_data["start"] = (self.start.row, self.start.col)
        if self.end:
            grid_data["end"] = (self.end.row, self.end.col)

        # 生成网格数据
        for row in self.grid:
            row_data = []
            for node in row:
                if node.is_start():
                    row_data.append('S')
                elif node.is_end():
                    row_data.append('E')
                elif node.is_barrier():
                    row_data.append('B')
                else:
                    row_data.append(' ')
            grid_data["grid"].append(''.join(row_data))

        # 写入JSON文件
        with open(args.file, "w") as f:
            json.dump(grid_data, f)
        print(f"网格已保存至{args.file}")

    def load_grid(self):
        try:
            with open(args.file, "r") as f:
                grid_data = json.load(f)

            # 清空当前网格
            for row in self.grid:
                for node in row:
                    node.reset()

            # 重建障碍物
            for r, row_str in enumerate(grid_data["grid"]):
                for c, char in enumerate(row_str):
                    node = self.grid[r][c]
                    if char == 'B':
                        node.make_barrier()
                    elif char == 'S':
                        node.make_start()
                        self.start = node
                        self.robot.reset_position(
                            self.start.x + self.start.width // 2,
                            self.start.y + self.start.width // 2)
                    elif char == 'E':
                        node.make_end()
                        self.end = node

            # 检查起点终点是否有效
            if grid_data["start"]:
                sr, sc = grid_data["start"]
                self.start = self.grid[sr][sc]
                self.start.make_start()
                self.robot.reset_position(self.start.x + self.start.width // 2,
                                          self.start.y + self.start.width // 2)
            if grid_data["end"]:
                er, ec = grid_data["end"]
                self.end = self.grid[er][ec]
                self.end.make_end()

            print(f"网格已加载自{args.file}")
        except Exception as e:
            print(f"加载失败: {str(e)}")

    def add_obstacle(self):
        if not self.route:
            print("目前没有路径可以被添加障碍物")
            print()
            return
        if self.route and not self.is_algorithm_running:
            # 在路径中随机添加障碍物
            if len(self.route) > 2:
                for _ in range(2):
                    obstacle_index = random.randint(0, len(self.route) - 1)
                    # obstacle_index = len(self.path) // 2
                    obstacle_node = self.route[obstacle_index]

                    if obstacle_node != self.start and obstacle_node != self.end:
                        # obstacle_node.make_barrier()
                        # 区分下颜色
                        obstacle_node.make_changed()
                        obstacle_node.g_score = float('inf')
                        obstacle_node.rhs = float('inf')
                        self.changed_nodes.append(obstacle_node)

    def analyze_performance(self):
        plt.close()
        plt.figure(1, figsize=(13, 8))
        # fig.suptitle('性能分析')

        # 迭代次数比较
        ax0 = plt.subplot2grid((3, 3), (0, 0))
        plt.xticks(fontsize=9)

        bar0 = ax0.bar(
            self.algorithms,
            [self.planning_iters.get(key, 0) for key in self.algorithms])
        add_bottom_labels(ax0, bar0)
        ax0.set_title('迭代次数比较')
        ax0.set_ylabel('次数')

        # 规划时间比较
        ax1 = plt.subplot2grid((3, 3), (0, 1))
        plt.xticks(fontsize=9)
        bar1 = ax1.bar(
            self.algorithms,
            [self.planning_times.get(key, 0) for key in self.algorithms])
        add_bottom_labels(ax1, bar1, 2)
        ax1.set_title('规划时间比较 (s)')
        ax1.set_ylabel('时间 (s)')

        # 规划路径长度比较
        ax2 = plt.subplot2grid((3, 3), (0, 2))
        plt.xticks(fontsize=9)
        bar2 = ax2.bar(
            self.algorithms,
            [self.path_lengths.get(key, 0) for key in self.algorithms])
        add_bottom_labels(ax2, bar2, 1)
        ax2.set_title('规划路径长度比较')
        ax2.set_ylabel('规划路径长度 (cost)')

        # 跟踪误差比较
        ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=3)

        for label, (values, _) in self.tracking_errors.items():
            x = np.linspace(0, self.path_length, len(values))
            ax3.plot(x, values, label=label)

        ax3.set_title('跟踪误差比较')
        ax3.set_xlabel('移动路程 (cost)')
        ax3.set_ylabel('与临时目标点的误差 (px)')
        if self.tracking_errors: ax3.legend()

        ax4 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
        for label, (_, values) in self.tracking_errors.items():
            x = np.linspace(0, self.path_length, len(values))
            ax4.plot(x, values, label=label)

        ax4.set_title('偏离路径误差比较')
        ax4.set_xlabel('移动路程 (cost)')
        ax4.set_ylabel('与规划路径的误差 (px)')
        if self.tracking_errors: ax4.legend()

        plt.tight_layout()
        plt.show()

    # 两个路径跟踪算法
    def running_tracking_algorithm(self):
        if not self.route:
            print("目前没有路径可以用于跟踪")
            print()
            return
        for i in self.changed_nodes:
            if i in self.route:
                print("原规划的路径上有新增障碍物，请重新运行路径规划算法或者手动去除障碍物")
                print()
                return
        if not (self.start and self.end and not self.is_algorithm_running):
            if not self.start: print("起点未设置")
            if not self.end: print("终点未设置")
            print()
            return
        if self.dstarlite:
            tracker = TRACKER(WIN, self.grid, self.rows, self.width,
                              self.is_show_grid, self.route, self.robot, VIS)
            self.start.make_closed()
            temp_start = self.start
            self.start = self.route[-1]
            self.start.make_start()
            errors = tracker.run_tracking(self.controller_type)

            while type(errors[0]) == int:
                if errors[0] == -1:
                    self.changed_nodes.append(errors[1])
                elif errors[0] == -2:
                    self.removed_nodes.append(errors[1])
                if errors[2].is_barrier():
                    errors[2].reset(isstrict=True)
                self.start = errors[2]
                self.robot.x, self.robot.y = tracker.robot.x, tracker.robot.y
                self.dstar_lite_replan(is_reset=False)
                tracker.route = self.route
                errors = tracker.run_tracking(self.controller_type,
                                              is_reset=False)
            self.start.make_closed()
            self.start = temp_start
            self.start.make_start()
            self.tracking_errors[self.controller_type] = errors
        else:
            tracker = TRACKER(WIN, self.grid, self.rows, self.width,
                              self.is_show_grid, self.route, self.robot, VIS)
            self.start.reset(isstrict=True)
            temp_start = self.start
            self.start = self.route[-1]
            self.start.make_start()
            errors = tracker.run_tracking(self.controller_type,
                                          can_operate=False)
            self.start.reset(isstrict=True)
            self.start = temp_start
            self.start.make_start()
            self.tracking_errors[self.controller_type] = errors

    def handle_button_events(self, event):
        mouse_pos = pygame.mouse.get_pos()

        if event.type == pygame.QUIT:
            self.isrun = False

        # 按钮悬停检测
        for button in VIS.buttons:
            button.check_hover(mouse_pos)

            # 算法选择
            if button.text.startswith("算法:") and button.is_clicked(
                    mouse_pos, event):
                if self.algorithm == "A*":
                    self.algorithm = "Dijkstra"
                    button.text = "算法: Dijkstra"
                elif self.algorithm == "D*lite":
                    self.algorithm = "A*"
                    button.text = "算法: A*"
                elif self.algorithm == "Dijkstra":
                    self.algorithm = "BFS"
                    button.text = "算法: BFS"
                else:
                    self.algorithm = "D*lite"
                    button.text = "算法: D*lite"

            # 控制器选择
            elif button.text.startswith("控制器:") and button.is_clicked(
                    mouse_pos, event):
                if self.controller_type == "PID":
                    self.controller_type = "MPC"
                    button.text = "控制器: MPC"
                else:
                    self.controller_type = "PID"
                    button.text = "控制器: PID"

            # 规划路径
            elif button.text == "规划路径" and button.is_clicked(mouse_pos, event):
                self.running_algorithm()
                # print()

            # 添加障碍物
            elif button.text == "在路径上添加障碍物" and button.is_clicked(
                    mouse_pos, event):
                self.add_obstacle()

            # 重规划
            elif button.text == "D*lite重规划" and button.is_clicked(
                    mouse_pos, event):
                self.dstar_lite_replan()
                # print()

            # 开始跟踪
            elif button.text == "开始跟踪" and button.is_clicked(mouse_pos, event):
                self.running_tracking_algorithm()

            # 性能分析
            elif button.text == "性能分析" and button.is_clicked(mouse_pos, event):
                self.analyze_performance()

            # 以下是一些个性化功能
            elif button.text == "显示网格" and button.is_clicked(mouse_pos, event):
                self.is_show_grid = not self.is_show_grid
                button.text = "不显示网格"
            elif button.text == "不显示网格" and button.is_clicked(
                    mouse_pos, event):
                self.is_show_grid = not self.is_show_grid
                button.text = "显示网格"
            elif button.text == "清除规划过程" and button.is_clicked(
                    mouse_pos, event):
                self.dstarlite = None
                VIS.reset_grid(self.grid)
            elif button.text == "清空" and button.is_clicked(mouse_pos, event):
                self.start = None
                self.end = None
                self.grid = VIS.make_grid(self.rows, self.width)
                self.route = []
                self.robot.trajectory = []
                self.dstarlite = None
            elif button.text == "导入地图" and button.is_clicked(mouse_pos, event):
                self.start = None
                self.end = None
                self.grid = VIS.make_grid(self.rows, self.width)
                self.route = []
                self.load_grid()
                self.dstarlite = None
            elif button.text == "导出地图" and button.is_clicked(mouse_pos, event):
                self.save_grid()

    def run(self):
        while self.isrun:
            self.clock.tick(60)

            # 事件处理
            mouse_pressed = pygame.mouse.get_pressed()
            if mouse_pressed[0] or mouse_pressed[2]:
                pos = pygame.mouse.get_pos()
                row, col = self.get_clicked_pos(pos)
                if 0 <= row < self.rows and 0 <= col < self.rows:  # 添加边界检查
                    node = self.grid[row][col]
                    if mouse_pressed[0]:  # 左键
                        if not self.start and node != self.end:
                            self.start = node
                            self.start.make_start()
                            self.robot.reset_position(
                                self.start.x + self.start.width // 2,
                                self.start.y + self.start.width // 2)
                        elif not self.end and node != self.start:
                            self.end = node
                            self.end.make_end()

                        elif node != self.end and node != self.start and not node.is_black_barrier(
                        ):
                            node.make_barrier()
                            self.changed_nodes.append(node)
                            if node in self.removed_nodes:
                                self.removed_nodes.remove(node)
                    else:  # 右键
                        if node.is_barrier():
                            self.removed_nodes.append(node)
                            if node in self.changed_nodes:
                                self.changed_nodes.remove(node)
                        node.reset(isstrict=True)
                        if node == self.start:
                            self.start = None
                            print("重置起点信息")
                            print()
                        elif node == self.end:
                            self.end = None
                            if self.dstarlite:
                                self.dstarlite = None
                                print("终点已改变")
                                print()

            for event in pygame.event.get():
                self.handle_button_events(event)

                if event.type == pygame.QUIT:
                    self.isrun = False

            if self.start:
                VIS.draw(WIN,
                         self.grid,
                         self.rows,
                         self.width,
                         self.is_show_grid,
                         self.route,
                         robot=self.robot)
            else:
                VIS.draw(WIN, self.grid, self.rows, self.width,
                         self.is_show_grid, self.route)
        pygame.quit()


if __name__ == "__main__":
    app = Application()
    app.run()
