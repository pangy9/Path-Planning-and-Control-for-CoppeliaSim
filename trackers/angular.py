import numpy as np
import math
from abc import ABC, abstractmethod
from utils import smooth_path_yaw
import matplotlib.pyplot as plt
# 设置matplotlib的字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class AngularController(ABC):
    """角度控制器抽象基类"""
    @abstractmethod
    def compute_angular_control(self, current_pose_px, path_xy, speed, **kwargs):
        """
        计算角度控制量
        返回: (omega, steering_angle) - 根据控制器类型，其中一个为None
        """
        pass
    
class PurePursuitAngularController(AngularController):
    """Pure Pursuit 路径跟踪算法实现"""
    def __init__(self, pixels_to_meters_scale, lookahead_distance=30):
        """
        Args:
            pixels_to_meters_scale (float): 像素到米的转换比例。
            lookahead_distance (float): 前瞻距离，单位为像素。
            WB (float): 车辆轴距 (Wheelbase)，用于计算转向角度。
        """
        self.lookahead_distance = lookahead_distance
        self.pixels_to_meters_scale = pixels_to_meters_scale

    def compute_angular_control(self, current_pose_px, path_xy, speed=None, return_steering=False, WB=None, display_image=None, **kwargs):
        if return_steering and WB is None:
            raise ValueError("需要返回转向角度，但未设置轴距 (WB)")
        
        current_pos = np.array(current_pose_px[:2])
        current_orientation = current_pose_px[2]
        
        lookahead_point, _ = self._find_lookahead_point(current_pos, path_xy)
        if lookahead_point is None: return 0, 0

        dx_px = lookahead_point[0] - current_pos[0]
        dy_px = lookahead_point[1] - current_pos[1]
        
        alpha = math.atan2(dy_px, dx_px) - current_orientation
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi

        lookahead_dist_px = np.hypot(dx_px, dy_px)
        lookahead_dist_m = lookahead_dist_px * self.pixels_to_meters_scale

        # 绘制当前路径点和目标点
        if display_image is not None:
            plt.cla()
            plt.imshow(display_image, origin='lower')
            plt.title(f"Pure Pursuit Path Tracking, lookahead_dist_px: {lookahead_dist_px:.2f}")
            plt.scatter(current_pos[0], current_pos[1], marker='x', c='b', s=30, label='当前位置')
            plt.scatter(lookahead_point[0], lookahead_point[1], marker='x', c='magenta', s=50, label='当前目标点')
            plt.plot(*zip(*path_xy), c='green', label='规划路径')
            plt.legend()
            plt.pause(0.01)


        if return_steering:
            delta = math.atan2(2.0 * WB * math.sin(alpha), lookahead_dist_m)
            return None, delta
        else:
            curvature = 2 * math.sin(alpha) / lookahead_dist_m
            omega = speed * curvature
            return omega, None

    def _find_lookahead_point(self, current_pos, path):
        if not path: return None, -1
        
        # 找到距离当前位置最近的路径点
        closest_idx = np.argmin([np.linalg.norm(current_pos - np.array(p)) for p in path])
        
        # 从最近点开始，向前搜索前瞻点
        for i in range(closest_idx, len(path)):
            if np.linalg.norm(current_pos - np.array(path[i])) >= self.lookahead_distance:
                return path[i], i
        
        # 如果没有找到满足前瞻距离的点，返回路径的最后一个点
        return path[-1], len(path) - 1

    
class StanleyAngularController(AngularController):
    """Stanley 控制器路径跟踪算法实现"""
    def __init__(self, pixels_to_meters_scale, WB=None, k_gain=0.5):
        """
        Args:
            pixels_to_meters_scale (float): 像素到米的转换比例。
            WB (float): 车辆轴距 (Wheelbase)，用于计算角速度。
            k_gain (float): Stanley 控制器的增益，用于横向误差的控制。
        """
        self.pixels_to_meters_scale = pixels_to_meters_scale
        self.k_gain = k_gain
        self.WB = WB  # 存储车辆轴距
        self._target_yaws = None  # 缓存路径点的目标航向角

    def _calculate_target_yaws(self, path_xy):
        """
        计算并缓存路径上每个点的目标航向角。
        """
        target_yaws = []
        n_points = len(path_xy)
        for i in range(n_points):
            # 寻找当前点前方的一个点用于计算方向
            # 如果是路径末尾，则使用最后一个路径段的方向
            p_ahead_idx = min(i + 1, n_points - 1)
            
            dx = path_xy[p_ahead_idx][0] - path_xy[i][0]
            dy = path_xy[p_ahead_idx][1] - path_xy[i][1]
            
            # 使用 atan2 计算航向角。
            target_yaw = math.atan2(dy, dx)
            target_yaws.append(target_yaw)
        return target_yaws

    def compute_angular_control(self, current_pose_px, path_xy, speed, return_steering=False, WB=None, display_image=None, **kwargs):
        if WB is not None:
            self.WB = WB
            
        # 如果路径更新了，重新计算所有点的目标航向角
        if self._target_yaws is None or len(self._target_yaws) != len(path_xy):
            self._target_yaws = smooth_path_yaw(self._calculate_target_yaws(path_xy))

        current_pos = np.array(current_pose_px[:2])
        current_orientation = current_pose_px[2]
        
        # 1. 找到路径上距离车辆最近的点
        target_idx, error_front_axle_px = self._find_closest_point_and_error(current_pose_px, path_xy)
        if target_idx is None:
            return speed, 0.0 # 如果找不到点，保持原速，不转向
        
        # 绘制当前路径点和目标点
        if display_image is not None:
            plt.cla()
            plt.imshow(display_image, origin='lower')
            plt.title(f"Stanley Control Tracking, 横向偏差(像素值): {error_front_axle_px:.2f}")
            plt.scatter(current_pos[0], current_pos[1], marker='x', c='b', s=30, label='当前位置')
            plt.scatter(path_xy[target_idx][0], path_xy[target_idx][1], marker='x', c='magenta', s=50, label='当前目标点')
            plt.plot(*zip(*path_xy), c='green', label='规划路径')
            plt.legend()
            plt.pause(0.01)
            
        # 2. 获取航向角误差 (theta_e)
        target_yaw = self._target_yaws[target_idx]
        heading_error = target_yaw - current_orientation
        
        # 将角度误差归一化到 [-pi, pi]
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi

        # 3. 获取横向误差 (e_fa)
        error_front_axle_m = error_front_axle_px * self.pixels_to_meters_scale
        
        # 4. 计算Stanley控制律，得到目标前轮转角 delta
        crosstrack_term = math.atan2(self.k_gain * error_front_axle_m, speed)
        print(f"当前航向角: {current_orientation:.2f}, 目标航向角: {target_yaw:.2f}, 横向误差(米): {error_front_axle_m:.2f}, 横向控制项: {crosstrack_term:.2f}")
        
        delta = heading_error + crosstrack_term
        
        # 限制转向角 - 这是必要的安全约束
        MAX_STEER = math.pi / 4
        delta = np.clip(delta, -MAX_STEER, MAX_STEER)
        
        if return_steering:
            return None, delta
        else:
            if self.WB is None:
                raise ValueError("需要计算角速度，但未设置轴距 (WB)")
            omega = speed * math.tan(delta) / self.WB
            return omega, None

    def _find_closest_point_and_error(self, current_pose, path):
        """
        找到路径上距离车辆前轮最近的点，并计算横向误差。认为车辆位置是车辆中心位置。
        """
        orientation = current_pose[2]
        current_pos = np.array(current_pose[:2])
        
        # Stanley控制基于前轴，不是车辆中心
        if self.WB is not None:
            front_axle_pos = current_pos + np.array([math.cos(orientation), math.sin(orientation)]) * self.WB / 2.0
        else:
            front_axle_pos = current_pos
        
        distances = [np.linalg.norm(front_axle_pos - np.array(p)) for p in path]
        closest_idx = np.argmin(distances)
        
        # 从前轴中心到最近路径点的向量
        vec_to_path_point = np.array(path[closest_idx]) - front_axle_pos
        
        # 计算路径在该点的切线方向
        if closest_idx < len(path) - 1:
            path_tangent = np.array(path[closest_idx + 1]) - np.array(path[closest_idx])
        else:
            path_tangent = np.array(path[closest_idx]) - np.array(path[closest_idx - 1])
        
        path_tangent_norm = path_tangent / (np.linalg.norm(path_tangent) + 1e-6)
        
        # 横向误差是向量在路径法线方向上的投影
        path_normal = np.array([-path_tangent_norm[1], path_tangent_norm[0]])
        error_front_axle = np.dot(vec_to_path_point, path_normal)
        
        return closest_idx, error_front_axle
        