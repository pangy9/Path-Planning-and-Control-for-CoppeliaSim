import numpy as np
from abc import ABC, abstractmethod
from utils import smooth
import heapq
import math
import random

class PathPlanner(ABC):
    """路径规划器的抽象基类 (接口)"""
    @abstractmethod
    def plan(self, start_xy, goal_xy, costmap):
        """
        规划从起点到终点的路径。
        使用数学坐标系，坐标系原点在左下角，x轴向右，y轴向上。

        Args:
            start_xy (tuple): 起点坐标(行, 列)。
            goal_xy (tuple): 终点坐标(行, 列)。
            costmap (np.ndarray): 一个二维数组，表示代价地图。障碍物应为 np.inf。

        Returns:
            list: 一个包含 (x, y) 坐标的路径列表，如果找不到路径则返回 None。
        """
        pass

class AStarPlanner(PathPlanner):
    """A* 路径规划算法实现, A*算法需要(r, c)坐标系"""
    def __init__(self, smooth_path=True, weight_data=0.001, weight_smooth=0.95):
        # 路径规划
        self.start_rc = None # 起点 (r, c)
        self.goal_rc = None # 终点 (r, c)
        self.G = None # 从起点A移动到指定方格的移动代价
        self.H = None # 从指定的方格移动到终点B的估算成本
        self.F = None # 总代价函数
        self.openset = set() # 开放列表
        self.openset = [] # 关闭列表
        self.openset_mirror = set() # 用于快速查找节点是否在开放列表中
        self.cameFrom = dict() # 记录路径父节点，从终点到起点
        self.costmap = None # world cost map: obstacle = Inf

        self.smooth_path = smooth_path
        self.weight_data = weight_data
        self.weight_smooth = weight_smooth

    def plan(self, start_xy, goal_xy, costmap):
        # 数学坐标系转换为数组索引（用于A*算法）
        start_x, start_y = start_xy
        goal_x, goal_y = goal_xy
        
        start_rc = (start_y, start_x)
        goal_rc = (goal_y, goal_x)
        self._initialize(start_rc, goal_rc, costmap)
        
        while self.openset:
            _, current = heapq.heappop(self.openset)
            if current is None or current == self.goal_rc:
                break
            
            self.closedset.add(current)
            
            # 获取8个方向的邻居
            for neighbor in self._get_neighbors(current):
                if neighbor in self.closedset or self.costmap[neighbor] == np.inf:
                    continue
                
                # 对角线移动的代价可以设为sqrt(2) * 1，水平垂直移动代价为1
                # 简化处理，所有有效移动代价都取 self.costmap[neighbor] (通常为1)
                new_g_cost = self.G[current] + self.costmap[neighbor]
                
                if neighbor not in self.openset_mirror or new_g_cost < self.G[neighbor]:
                    self.cameFrom[neighbor] = current
                    self.G[neighbor] = new_g_cost
                    self.H[neighbor] = self._heuristic(neighbor, self.goal_rc)
                    self.F[neighbor] = self.G[neighbor] + self.H[neighbor]
                    heapq.heappush(self.openset, (self.F[neighbor], neighbor))
                    self.openset_mirror.add(neighbor)
                            
        if self.goal_rc not in self.cameFrom:
            print("A* 路径未找到")
            return None

        print("A* 路径规划完成")
        path_rc = self._reconstruct_path()
        path_xy = [(c, r) for r, c in path_rc]
        
        if self.smooth_path:
            print("进行平滑处理")
            return smooth(path_xy, weight_data=self.weight_data, weight_smooth=self.weight_smooth)
        else:
            return path_xy

    def _initialize(self, start_rc, goal_rc, costmap):
        self.costmap, self.start_rc, self.goal_rc = costmap, start_rc, goal_rc
        resolution = costmap.shape
        self.G = np.full(resolution, np.inf, dtype=np.float32)
        self.H = np.full(resolution, np.inf, dtype=np.float32)
        self.F = np.full(resolution, np.inf, dtype=np.float32)
        self.openset = [(self.F[start_rc], start_rc)]
        self.openset_mirror = {start_rc}        
        self.closedset = set()
        self.cameFrom = {start_rc: None}
        self.G[start_rc] = 0
        self.H[start_rc] = self._heuristic(start_rc, goal_rc)
        self.F[start_rc] = self.G[start_rc] + self.H[start_rc]

    def _reconstruct_path(self):
        path = []
        current = self.goal_rc
        while current is not None:
            path.append(current)
            current = self.cameFrom.get(current)
        return path[::-1]

    def _get_neighbors(self, node):
        r, c = node
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        r_max, c_max = self.costmap.shape
        return [(r + dr, c + dc) for dr, dc in directions if 0 <= r + dr < r_max and 0 <= c + dc < c_max]

    def _heuristic(self, start, goal):
        return abs(start[0] - goal[0]) + abs(start[1] - goal[1])


class RRTPlanner(PathPlanner):
    """RRT 路径规划算法实现"""
    class Node:
        """
        RRT Node
        """
        def __init__(self, position):
            self.x = position[0]
            self.y = position[1]
            self.path_x = []
            self.path_y = []
            self.parent = None
            
    def __init__(
        self,
        expand_dis=3.0,
        path_resolution=1,
        goal_sample_rate=5,
        max_iter=5000,
        smooth_path=True,
        weight_data=0.3,
        weight_smooth=0.8,
    ):
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        
        # 平滑路径的参数
        self.smooth_path = smooth_path
        self.weight_data = weight_data
        self.weight_smooth = weight_smooth

    def plan(self, start_xy, goal_xy, costmap):
        height, width = costmap.shape
        
        node_list = [self.Node(start_xy)]
        for i in range(self.max_iter):
            if random.randint(0, 100) > self.goal_sample_rate:
                rnd_node = self.Node((random.uniform(0, width), random.uniform(0, height)))
            else:  # goal point sampling
                rnd_node = self.Node(goal_xy)
            nearest_ind = self.get_nearest_node_index(node_list, rnd_node)
            nearest_node = node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_collision(new_node, costmap):
                node_list.append(new_node)

            if np.linalg.norm(np.array([node_list[-1].x, node_list[-1].y]) - np.array(goal_xy)) <= self.expand_dis:
                final_node = self.steer(node_list[-1], self.Node(goal_xy), self.expand_dis)
                if self.check_collision(final_node, costmap):
                    node_list.append(final_node)
                    path = self.generate_final_course(node_list)
                    print("RRT 路径规划完成")
                    if self.smooth_path:
                        print("进行平滑处理")
                        return smooth(path, weight_data=self.weight_data, weight_smooth=self.weight_smooth)
                    else:
                        return path
        
        print("RRT 未找到路径")
        return None
    
    def get_nearest_node_index(self, node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2
                 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    def steer(self, from_node, to_node, extend_length=float("inf")):

        new_node = self.Node((from_node.x, from_node.y))
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.x = to_node.x
            new_node.y = to_node.y

        new_node.parent = from_node

        return new_node
    
    def check_collision(self, node, costmap):

        if node is None:
            return False

        for x, y in zip(node.path_x, node.path_y):
            if costmap[int(y), int(x)] == np.inf:
                return False
        
        return True

    def calc_distance_and_angle(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

    def generate_final_course(self, node_list):
        path = [[node_list[-1].x, node_list[-1].y]]
        node = node_list[-2]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        
        path.reverse()
        return path

class RRTStarPlanner(RRTPlanner):
    """RRT* 路径规划算法实现"""
    class Node(RRTPlanner.Node):
        def __init__(self, position):
            super().__init__(position)
            self.cost = 0.0
        
    def __init__(
        self,
        expand_dis=3.0,
        path_resolution=1,
        goal_sample_rate=5,
        max_iter=5000,
        smooth_path=True,
        weight_data=0.3,
        weight_smooth=0.8,
        connect_circle_dist=50.0,  # 连接圆的半径
    ):
        super().__init__(
            expand_dis=expand_dis,
            path_resolution=path_resolution,
            goal_sample_rate=goal_sample_rate,
            max_iter=max_iter,
            smooth_path=smooth_path,
            weight_data=weight_data,
            weight_smooth=weight_smooth,
        )
        self.connect_circle_dist = connect_circle_dist

    def plan(self, start_xy, goal_xy, costmap):
        height, width = costmap.shape

        node_list = [self.Node(start_xy)]
        goal_node = self.Node(goal_xy)
        for i in range(self.max_iter):
            if random.randint(0, 100) > self.goal_sample_rate:
                rnd_node = self.Node((random.uniform(0, width), random.uniform(0, height)))
            else:  # goal point sampling
                rnd_node = self.Node(goal_xy)
            nearest_ind = self.get_nearest_node_index(node_list, rnd_node)
            new_node = self.steer(node_list[nearest_ind], rnd_node, self.expand_dis)
            near_node = node_list[nearest_ind]
            new_node.cost = near_node.cost + math.hypot(new_node.x-near_node.x, new_node.y-near_node.y)

            if self.check_collision(new_node, costmap):
                near_nodes = self.find_near_nodes(node_list, new_node)
                node_with_updated_parent = self.choose_parent(new_node, near_nodes, costmap)
                if node_with_updated_parent:
                    self.rewire(node_list, node_with_updated_parent, near_nodes, costmap)
                    node_list.append(node_with_updated_parent)
                else:
                    node_list.append(new_node)

            if new_node:
                last_index = self.search_best_goal_node(node_list, goal_node, costmap)
                if last_index is not None:
                    node_list.append(goal_node)
                    path = self.generate_final_course(node_list)
                    print("RRT* 路径规划完成")
                    if self.smooth_path:
                        print("进行平滑处理")
                        return smooth(path, weight_data=self.weight_data, weight_smooth=self.weight_smooth)
                    else:
                        return path

        last_index = self.search_best_goal_node(node_list, goal_node, costmap)
        if last_index is not None:
            node_list.append(goal_node)
            path = self.generate_final_course(node_list)
            print("RRT* 路径规划完成")
            if self.smooth_path:
                print("进行平滑处理")
                return smooth(path, weight_data=self.weight_data, weight_smooth=self.weight_smooth)
            else:
                return path
        else:
            print("RRT* 未找到路径")
            return None

    def find_near_nodes(self, node_list, new_node):
        """
        1) defines a ball centered on new_node
        2) Returns all nodes of the three that are inside this ball
        """
        nnode = len(node_list) + 1
        r = self.connect_circle_dist * math.sqrt(math.log(nnode) / nnode)
        if hasattr(self, 'expand_dis'):
            r = min(r, self.expand_dis)
        near_nodes = [node for node in node_list if (node.x - new_node.x)**2 + (node.y - new_node.y)**2 <= r**2]
        return near_nodes

    def choose_parent(self, new_node, near_nodes, costmap):
        """
        Computes the cheapest point to new_node contained in the list
        near_nodes and set such a node as the parent of new_node.
        """
        if not near_nodes:
            return None

        # search nearest cost in near_nodes
        costs = []
        for near_node in near_nodes:
            t_node = self.steer(near_node, new_node)
            if t_node and self.check_collision(t_node, costmap):
                costs.append(self.calc_distance_and_angle(near_node, t_node)[0] + near_node.cost)
            else:
                costs.append(float("inf"))  # the cost of collision node
        min_cost = min(costs)

        if min_cost == float("inf"):
            print("There is no good path.(min_cost is inf)")
            return None

        min_node = near_nodes[costs.index(min_cost)]
        new_node = self.steer(min_node, new_node)
        new_node.cost = min_cost

        return new_node
    
    def rewire(self, node_list, new_node, near_nodes, costmap):
        """
            For each node in near_nodes, this will check if it is cheaper to
            arrive to them from new_node.
            In such a case, this will re-assign the parent of the nodes in
            near_inds to new_node.

        """
        for i, near_node in enumerate(near_nodes):
            edge_node = self.steer(new_node, near_node)
            if not edge_node:
                continue
            edge_node.cost = self.calc_distance_and_angle(near_node, new_node)[0] + near_node.cost

            no_collision = self.check_collision(edge_node, costmap)
            improved_cost = near_node.cost > edge_node.cost

            if no_collision and improved_cost:
                for node in self.node_list:
                    if node.parent == node_list[i]:
                        node.parent = edge_node
                node_list[i] = edge_node
                self.propagate_cost_to_leaves(node_list[i])
                
    def propagate_cost_to_leaves(self, node_list, parent_node):

        for node in node_list:
            if node.parent == parent_node:
                node.cost = self.calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node_list, node)
                          
    def search_best_goal_node(self, node_list, goal_node, costmap):
        dist_to_goal_list = [
            math.hypot(node.x - goal_node.x, node.y - goal_node.y) for node in node_list
        ]
        
        goal_inds = [
            dist_to_goal_list.index(i) for i in dist_to_goal_list
            if i <= self.expand_dis
        ]

        safe_goal_inds = []
        for goal_ind in goal_inds:
            t_node = self.steer(node_list[goal_ind], goal_node)
            if self.check_collision(t_node, costmap):
                safe_goal_inds.append(goal_ind)

        if not safe_goal_inds:
            return None

        safe_goal_costs = [node_list[i].cost + math.hypot(node_list[i].x - goal_node.x, node_list[i].y - goal_node.y)
                           for i in safe_goal_inds]

        min_cost = min(safe_goal_costs)
        for i, cost in zip(safe_goal_inds, safe_goal_costs):
            if cost == min_cost:
                return i

        return None