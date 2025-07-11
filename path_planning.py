import pygame
import time
import heapq


# 启发式函数
def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    # return abs(x1 - x2) + abs(y1 - y2)
    # return ((x1 - x2)**2 + (y1 - y2)**2)**0.5
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    return (dx + dy) + (2**0.5 - 2) * min(dx, dy)


# 重建路径
def reconstruct_path(came_from, current, draw):
    accessible_route = [current]
    while current in came_from:
        current = came_from[current]
        accessible_route.append(current)
        if not current.is_start() and not current.is_end():
            current.make_path()
        draw()
    return accessible_route


# A* 算法，后续的Dijkstra和BFS代码结构与之相似
def Astar_algorithm(draw, start, end):
    # 记录运行时长
    start_time = time.time()
    iteration_count = 1
    start.g_score = 0
    start.h_score = h(start.get_pos(), end.get_pos())
    open_list = [[start.h_score, start]]

    came_from = {}

    while open_list:
        # 可中途暂停算法
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("manual quit")
                exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                print("manual escape")
                return iteration_count, time.time() - start_time, []

        # 获取F值最小的节点
        current_f_score, current = min(open_list)
        open_list.remove([current_f_score, current])
        if current != start:
            current.make_closed()

        if current == end:
            accessible_route = reconstruct_path(came_from, end, draw)
            end.make_end()
            print('A* cost: ', current_f_score)
            return iteration_count, time.time() - start_time, accessible_route

        for neighbor in current.neighbors:

            # 计算移动代价
            dist = 1.5 if abs(neighbor.row -
                              current.row) + abs(neighbor.col -
                                                 current.col) == 2 else 1
            temp_g_score = current.g_score + dist

            # 比neighbor原来的G值小，则更新neighbor
            if temp_g_score < neighbor.g_score:
                came_from[neighbor] = current
                neighbor.g_score = temp_g_score
                idx = next((i for i in range(len(open_list))
                            if open_list[i][-1] == neighbor), None)

                if not idx:
                    # 如果neighbor不在openlist则加入openlist
                    neighbor.h_score = h(neighbor.get_pos(), end.get_pos())
                    open_list.append(
                        [temp_g_score + neighbor.h_score, neighbor])
                    neighbor.make_open()
                else:
                    # 如果在则直接更新G值
                    open_list[idx][0] = temp_g_score + neighbor.h_score

        # 更新画面
        draw()
        iteration_count += 1

    return iteration_count, time.time() - start_time, False


def Dijkstra_algorithm(draw, start, end):
    start_time = time.time()
    iteration_count = 1
    start.g_score = 0
    open_list = [[0, start]]

    came_from = {}

    while open_list:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("manual quit")
                exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                print("manual escape")
                return iteration_count, time.time() - start_time, []

        current_g_score, current = min(open_list)
        open_list.remove([current_g_score, current])
        if current != start:
            current.make_closed()

        if current == end:
            accessible_route = reconstruct_path(came_from, end, draw)
            end.make_end()
            print('Dijkstra cost: ', current_g_score)
            return iteration_count, time.time() - start_time, accessible_route

        for neighbor in current.neighbors:

            dist = 1.5 if abs(neighbor.row -
                              current.row) + abs(neighbor.col -
                                                 current.col) == 2 else 1
            temp_g_score = current.g_score + dist

            if temp_g_score < neighbor.g_score:
                came_from[neighbor] = current
                neighbor.g_score = temp_g_score
                idx = next((i for i in range(len(open_list))
                            if open_list[i][-1] == neighbor), None)
                if not idx:
                    open_list.append([neighbor.g_score, neighbor])
                    neighbor.make_open()
                else:
                    open_list[idx][0] = temp_g_score

        draw()
        iteration_count += 1

    return iteration_count, time.time() - start_time, False


def BFS(draw, start, end):
    start_time = time.time()
    iteration_count = 1
    start.g_score = 0
    open_list = [[0, start]]

    came_from = {}

    while open_list:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("manual quit")
                exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                print("manual escape")
                return iteration_count, time.time() - start_time, []

        current_g_score, current = open_list[0]
        open_list.remove([current_g_score, current])
        if current != start:
            current.make_closed()

        if current == end:
            accessible_route = reconstruct_path(came_from, end, draw)
            end.make_end()
            print('BFS cost: ', current_g_score)
            return iteration_count, time.time() - start_time, accessible_route

        for neighbor in current.neighbors:

            dist = 1.5 if abs(neighbor.row -
                              current.row) + abs(neighbor.col -
                                                 current.col) == 2 else 1
            temp_g_score = current.g_score + dist

            if temp_g_score < neighbor.g_score:
                came_from[neighbor] = current
                neighbor.g_score = temp_g_score
                idx = next((i for i in range(len(open_list))
                            if open_list[i][-1] == neighbor), None)
                if not idx:
                    open_list.append([neighbor.g_score, neighbor])
                    neighbor.make_open()
                else:
                    open_list[idx][0] = temp_g_score

        draw()
        iteration_count += 1

    return iteration_count, time.time() - start_time, False


class Dstar_lite:

    def __init__(self, grid, start, end, draw=None):
        self.grid = grid
        self.start = start
        self.end = end
        self.draw = draw
        self.open_list = []
        self.km = 0  # 路径成本偏移量
        self.initial_plan_complete = False
        self.camefrom = {}

        # 初始化节点
        for row in grid:
            for node in row:
                node.g_score = float('inf')
                node.rhs = float('inf')
                node.key = None

        # 设置终点
        self.end.rhs = 0
        self.end.key = self.calculate_key(self.end)
        heapq.heappush(self.open_list, (self.end.key, self.end))

    def calculate_key(self, node):
        k1 = min(node.g_score, node.rhs) + h(node.get_pos(),
                                             self.start.get_pos()) + self.km
        k2 = min(node.g_score, node.rhs)
        return (k1, k2)

    def update_vertex(self, node):
        if node != self.end:
            # 计算最小rhs值
            min_rhs = float('inf')
            min_neighbor = None

            for neighbor in node.neighbors:
                # 计算移动代价
                dist = 1.5 if abs(neighbor.row -
                                  node.row) + abs(neighbor.col -
                                                  node.col) == 2 else 1
                cost = neighbor.g_score + dist

                if cost < min_rhs:
                    min_rhs = cost
                    min_neighbor = neighbor

            node.rhs = min_rhs
            self.camefrom[node] = min_neighbor

        # 从open_list中移除节点（如果存在）
        for i, (key, n) in enumerate(self.open_list):
            if n == node:
                del self.open_list[i]
                heapq.heapify(self.open_list)
                if node != self.end:
                    node.make_closed()
                break

        # 如果g和rhs不一致，重新加入open_list
        if node.g_score != node.rhs:
            node.key = self.calculate_key(node)
            heapq.heappush(self.open_list, (node.key, node))
            if node != self.end:
                node.make_open()

    def process_state(self):
        if not self.open_list:
            return -1

        # 获取最小key的节点
        k_old, current = heapq.heappop(self.open_list)
        if current != self.end:
            current.make_closed()

        # 检查事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("manual quit")
                exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                print("manual escape")
                return -2

        # 更新节点状态
        if k_old < self.calculate_key(current):
            heapq.heappush(self.open_list,
                           (self.calculate_key(current), current))
            if current != self.end:
                current.make_open()
        elif current.g_score > current.rhs:
            current.g_score = current.rhs
            for neighbor in current.neighbors:
                self.update_vertex(neighbor)
        else:
            current.g_score = float('inf')
            self.update_vertex(current)
            for neighbor in current.neighbors:
                self.update_vertex(neighbor)

        if self.draw:
            self.draw()

        return 1

    def init_plan(self):
        start_time = time.time()
        iteration_count = 1

        while True:
            result = self.process_state()
            if result == -1:  # 无更多节点
                return iteration_count, time.time() - start_time, False
                # break
            if result == -2:  # 用户中断
                return iteration_count, time.time() - start_time, []

            iteration_count += 1

            # 检查起点是否稳定
            if self.start.g_score == self.start.rhs and self.start.g_score != float(
                    'inf'):
                break

        self.start.make_start()
        self.initial_plan_complete = True
        print('D*lite cost: ', self.start.g_score)
        return iteration_count, time.time() - start_time, True

    def add_obstacle(self, nodes):
        for node in nodes:
            self.camefrom.pop(node, None)
            # 从open_list中移除节点（如果存在）
            self.open_list[:] = (x for x in self.open_list if x[-1] != node)

        for node in nodes:
            # 更新邻居节点
            for neighbor in node.neighbors:
                if neighbor.is_barrier():
                    continue

                if self.camefrom.get(neighbor) == node:
                    self.update_vertex(neighbor)

    def remove_obstacle(self, nodes):
        for node in nodes:
            self.update_vertex(node)

        for node in nodes:
            # 更新邻居节点
            for neighbor in node.neighbors:
                if neighbor.is_barrier():
                    continue
                self.update_vertex(neighbor)

    def replan(self):
        if not self.initial_plan_complete:
            return 0, 0, False

        start_time = time.time()
        iteration_count = 1

        # 添加所有受影响节点到open_list
        for row in self.grid:
            for node in row:
                if node.g_score != node.rhs and not node.is_barrier():
                    node.key = self.calculate_key(node)
                    heapq.heappush(self.open_list, (node.key, node))
                    if node != self.end:
                        node.make_open()

        while True:
            # 检查起点是否稳定
            if self.start.g_score == self.start.rhs and self.start.g_score != float(
                    'inf') and self.open_list[0][0] >= self.calculate_key(
                        self.start):
                break
            result = self.process_state()
            if result == -1:  # 无更多节点
                return iteration_count, time.time() - start_time, False
                # break
            if result == -2:  # 用户中断
                return iteration_count, time.time() - start_time, []

            iteration_count += 1

        self.start.make_start()
        print('D*lite replan cost: ', self.start.g_score)
        return iteration_count, max(1e-12, time.time() - start_time), True

    def reconstruct_path(self):
        accessible_route = reconstruct_path(self.camefrom, self.start,
                                            self.draw)
        return accessible_route[::-1]
