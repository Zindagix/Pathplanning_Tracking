import pygame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


def add_bottom_labels(ax, bars, roundpoint=0):
    # 在柱子底部添加标注
    for bar in bars:
        height = bar.get_height()
        if not height: continue
        ax.annotate(
            f'{round(height,roundpoint)}',
            xy=(bar.get_x() + bar.get_width() / 2, 0),  # 标注点位置
            xytext=(0, 15),  # 文本偏移量（15px）
            textcoords="offset points",
            ha='center',
            va='top',  # 文本顶部对齐标注点
            fontsize=10,
            color='black')


# 颜色定义
COLORS = {
    "point": (235, 235, 235),
    "robot": (70, 130, 180),
    "changed": (82, 64, 64),
    "text": (30, 30, 30),
    "button": (153, 225, 255),
    "button_hover": (65, 175, 255),
    "line": (100, 200, 255),
    "robot": (0, 100, 255),
    "RED": (255, 50, 50),
    "PINK": (255, 125, 125),
    "BLUE": (51, 51, 255),
    "YELLOW": (255, 105, 255),
    "WHITE": (255, 255, 255),
    "BLACK": (0, 0, 0),
    "SKYBLUE": (125, 125, 255),
    "ORANGE": (255, 165, 0),
    "GREY": (128, 128, 128),
    "TURQUOISE": (64, 224, 255),
}


class Node:

    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = COLORS["WHITE"]
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows
        self.total_cols = total_rows
        self.g_score = float("inf")
        self.h_score = None
        self.rhs = None

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == COLORS["RED"]

    def is_open(self):
        return self.color == COLORS["PINK"]

    def is_barrier(self):
        return self.color == COLORS["BLACK"] or self.color == COLORS["changed"]

    def is_black_barrier(self):
        return self.color == COLORS["BLACK"]

    def is_start(self):
        return self.color == COLORS["ORANGE"]

    def is_end(self):
        return self.color == COLORS["TURQUOISE"]

    def reset(self, isstrict=False):
        if not self.is_start() and \
            not self.is_end() and not self.is_barrier():
            self.color = COLORS["WHITE"]
        elif isstrict:
            self.color = COLORS["WHITE"]
        self.g_score = float("inf")
        self.h_score = None
        self.neighbors = []
        self.rhs = float('inf')
        self.key = None

    def make_start(self):
        self.color = COLORS["ORANGE"]

    def make_closed(self):
        self.color = COLORS["RED"]

    def make_open(self):
        self.color = COLORS["PINK"]

    def make_barrier(self):
        self.color = COLORS["BLACK"]

    def make_end(self):
        self.color = COLORS["TURQUOISE"]

    def make_path(self):
        self.color = COLORS["SKYBLUE"]

    def make_changed(self):
        self.color = COLORS["changed"]

    def draw(self, win):
        pygame.draw.rect(win, self.color,
                         (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid):
        self.neighbors = []

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1),
                      (-1, 1), (-1, -1)]

        for dr, dc in directions:
            r, c = self.row + dr, self.col + dc
            if 0 <= r < self.total_rows and 0 <= c < self.total_cols and not grid[
                    r][c].is_barrier():
                self.neighbors.append(grid[r][c])

    def __lt__(self, other):
        return False


# 按钮类
class Button:

    def __init__(self, x, y, width, height, text):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = COLORS["button"]
        self.hover_color = COLORS["button_hover"]
        self.text_color = COLORS["text"]
        self.font = pygame.font.SysFont('SimHei', 28)
        self.hovered = False

    def draw(self, win):
        color = self.hover_color if self.hovered else self.color
        pygame.draw.rect(win, color, self.rect, border_radius=5)
        pygame.draw.rect(win, COLORS['GREY'], self.rect, 2, border_radius=5)

        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        win.blit(text_surface, text_rect)

    def check_hover(self, pos):
        self.hovered = self.rect.collidepoint(pos)

    def is_clicked(self, pos, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self.rect.collidepoint(pos)
        return False


class Visualization:

    def __init__(self, width, offset):
        self.width = width
        self.buttons = self.create_buttons()
        self.offset = offset

        self.fig = plt.figure(figsize=(6.6, 5.2))
        self.ax0 = self.fig.add_subplot(211)

        self.ax1 = self.fig.add_subplot(212)
        self.surf = None
        self.count = 0

        self.last_rect = pygame.Rect(0, 0, 0, 0)
        self.robot_rect = pygame.Rect(0, 0, 0, 0)

    # 创建网格
    def make_grid(self, rows, width):
        grid = []
        gap = width // rows
        for i in range(rows):
            grid.append([])
            for j in range(rows):
                node = Node(i, j, gap, rows)
                grid[i].append(node)
        return grid

    # 绘制网格线
    def draw_grid(self, win, rows, width):
        gap = width // rows
        for i in range(rows):
            pygame.draw.line(win, COLORS["GREY"], (0, i * gap),
                             (width, i * gap))
            for j in range(rows):
                pygame.draw.line(win, COLORS["GREY"], (j * gap, 0),
                                 (j * gap, width))

    def reset_grid(self, grid):
        for row in grid:
            for node in row:

                node.reset()

    def create_buttons(self):
        buttons = []
        x_pos = self.width + 20
        width = 320
        height = 45
        spacing = 62

        buttons_lists = [
            "算法: D*lite", "规划路径", "在路径上添加障碍物", "D*lite重规划", "控制器: PID", "开始跟踪",
            "性能分析"
        ]

        for i, buttons_target in enumerate(buttons_lists):
            # 算法选择按钮
            buttons.append(
                Button(x_pos, 20 + spacing * i, width, height, buttons_target))

        buttons_lists = ["导入地图", "导出地图", "显示网格", "清除规划过程", "清空"]

        for i, buttons_target in enumerate(buttons_lists):
            # 算法选择按钮
            buttons.append(
                Button(x_pos + 340, 20 + spacing * i, width, height,
                       buttons_target))

        return buttons

    def draw_route(self, win, route):

        for i in range(len(route) - 1):
            start_pos = (route[i].x + route[i].width // 2,
                         route[i].y + route[i].width // 2)
            end_pos = (route[i + 1].x + route[i + 1].width // 2,
                       route[i + 1].y + route[i + 1].width // 2)
            pygame.draw.line(win, COLORS["BLUE"], start_pos, end_pos, 4)

    def draw(self, win, grid, rows, width, isgrid=True, route=[], robot=None):
        # 绘制整个画面

        for row in grid:
            for node in row:
                node.draw(win)
        if isgrid:
            self.draw_grid(win, rows, width)

        # 绘制侧边栏背景
        pygame.draw.rect(win, (220, 220, 220),
                         (self.width, 0, self.offset, self.width))

        for button in self.buttons:
            button.draw(win)
        if route:
            self.draw_route(win, route)
        if robot:
            self.draw_robot(win, robot)

        self.draw_text(win, 10, 10, "按 Esc 暂停算法")

        pygame.display.update()

    def draw_robot(self, win, robot):
        # 绘制机器人
        x, y, theta = robot.x, robot.y, robot.theta

        # 绘制机器人主体
        pygame.draw.circle(win, COLORS["robot"], (int(x), int(y)), 12)

        # 绘制方向指示器
        end_x = x + 16 * np.cos(theta)
        end_y = y + 16 * np.sin(theta)
        pygame.draw.line(win, COLORS["BLACK"], (x, y), (end_x, end_y), 7)
        pygame.draw.line(win, COLORS["point"], (x, y), (end_x, end_y), 3)

        # 绘制轨迹
        if len(robot.trajectory) > 1:
            points = [(p[0], p[1]) for p in robot.trajectory]
            pygame.draw.lines(win, COLORS["line"], False, points, 2)

    def draw_tracking(self, win, grid, rows, width, isgrid, route, robot,
                      target_x, target_y):
        self.robot_rect = pygame.Rect(robot.x - 40, robot.y - 40, 80, 80)

        for row in grid:
            for node in row:
                if self.robot_rect.collidepoint(node.x, node.y):
                    node.draw(win)
                elif self.last_rect.collidepoint(node.x, node.y):
                    node.draw(win)
        if isgrid:
            self.draw_grid(win, rows, width)

        # 绘制侧边栏背景
        # pygame.draw.rect(win, (220, 220, 220),
        #                  (self.width, 0, self.offset, self.width))

        # for button in self.buttons:
        #     button.draw(win)

        self.draw_circle(win, target_x, target_y)
        if route:
            self.draw_route(win, route)
        if robot:
            self.draw_robot(win, robot)

        self.draw_text(win, 10, 10, "按 Esc 暂停算法")

    def draw_circle(self, win, x, y):
        # 绘制当前目标点
        pygame.draw.circle(win, COLORS["YELLOW"], (x, y), 10)

    def draw_errortext(self, win, x, y, control_method, error):
        font = pygame.font.SysFont('SimHei', 24)
        text = font.render(f"控制方法: {control_method} | 跟踪误差: {error:.2f} px",
                           True, COLORS["GREY"])
        win.blit(text, (x, y))
        if x + text.get_rect().width > self.last_rect.width:
            self.last_rect = pygame.Rect(0, 0, x + text.get_rect().width,
                                         y + text.get_rect().height)
        pygame.display.update(self.last_rect)
        pygame.display.update(self.robot_rect)

    def draw_text(self, win, x, y, text, size=24):
        font = pygame.font.SysFont('SimHei', size)
        text = font.render(f"{text}", True, COLORS["GREY"])
        win.blit(text, (x, y))

    def draw_dynamic_errors(self, win, errors, errors2paths, control_method):

        # if self.count % 1 == 0 or control_method != 'PID':
        self.ax0.clear()
        self.ax0.plot(errors, color='purple')
        self.ax0.set_title('跟踪误差')
        self.ax0.set_xlabel('时间步 (step)')
        self.ax0.set_ylabel('与临时目标点的误差 (px)')

        self.ax1.clear()
        self.ax1.plot(errors2paths, color='skyblue')
        self.ax1.set_title('偏离路径误差')
        self.ax1.set_xlabel('时间步 (step)')
        self.ax1.set_ylabel('与规划路径的误差 (px)')

        self.fig.tight_layout()

        canvas = FigureCanvasAgg(self.fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        size = canvas.get_width_height()
        self.surf = pygame.image.frombuffer(buf, size, 'RGBA')
        # self.count += 1
        win.blit(self.surf, (self.width + 20, 456))
        pygame.display.update(pygame.Rect((self.width + 20, 456), size))
