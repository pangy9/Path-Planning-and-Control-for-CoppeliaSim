"""
Original Author: Long Cheng
Date: 2024-05-11 15:30:19
LastEditors: Long Cheng
LastEditTime: 2024-06-07 16:25:18
Description: 

Copyright (c) 2024 by longcheng.work@outlook.com, All Rights Reserved.

Modified by: Yun Pang
Modifications: Add new functions, like plot_summary_results..., and modified __init__, estimate_robot_sim_energy() functions.
Used with permission from original author / Educational use only
"""

"""

坐标系统一使用数学坐标系
坐标系原点在图像左下角，y轴向上，x轴向右
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class Reporter:
    def __init__(self, args=None, config_details=None, debug_plot=True, pixels_to_meters_scale=0.1):
        self._debug_plot = debug_plot
        
        # 根据传入的参数生成文件名和报告头
        if args:
            config_name = f"{args.vehicle_model}_{args.planner}_{args.angular}_{args.speed}"
            self._report_str = f"--- 实验配置 ---\n"
            self._report_str += f"车辆模型: {args.vehicle_model}\n"
            self._report_str += f"路径规划器: {args.planner}\n"
            self._report_str += f"角度控制器: {args.angular}\n"
            self._report_str += f"速度控制器: {args.speed}\n"
        else:
            config_name = "default_config"
            self._report_str = "--- 实验配置: 默认 ---\n"

        if config_details:
            self._report_str += self._format_config_details(config_details)

        # 创建报告目录
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        report_dir = f"./report/{timestamp}_{config_name}"
        self._report_file_base = f"{report_dir}/{timestamp}_{config_name}"
        os.makedirs(report_dir, exist_ok=True)
        
        self._report_file = f"{self._report_file_base}.txt"

        # ... (其余初始化变量保持不变) ...
        self._init_image = None
        self._obstacle_mask = None
        self._plan_path = None
        self._start_position = None
        self._goal_position = None
        self._robot_sim_path = np.empty(shape=(0, 2))
        self._robot_sim_orientation = np.empty(shape=(0, 1))
        self._robot_sim_time = np.empty(shape=(0, 1))
        self.__goal_tolerance = 10
        self.__inflate_size = 30
        self._max_speed = 5
        self._max_rotation_speed = 5
        self.pixels_to_meters_scale = pixels_to_meters_scale
        
        
    def _format_config_details(self, config_details):
        """将详细的配置字典格式化为字符串"""
        details_str = "\n--- 详细参数 ---\n"
        for component, params in config_details.items():
            details_str += f"[{component.replace('_', ' ').title()}]\n"
            if not params:
                details_str += "  (无额外参数)\n"
                continue
            for key, value in params.items():
                if isinstance(value, np.ndarray):
                    value_str = np.array2string(value.flatten(), precision=3, separator=', ')
                else:
                    value_str = str(value)
                details_str += f"  {key}: {value_str}\n"
        details_str += "------------------\n\n"
        return details_str
    
    def add(self, message):
        self._report.append(message)

    @staticmethod
    def check_rgb_image(image):
        # check if image is a numpy array
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy array")
        # check if image is a 3D array
        if image.ndim != 3:
            raise ValueError("image must be a 3D array")
        # check if image has 3 channels
        if image.shape[2] != 3:
            raise ValueError("image must have 3 channels")
        return True

    @staticmethod
    def check_image_mask(image_mask):
        # check if image_mask is a numpy array
        if not isinstance(image_mask, np.ndarray):
            raise TypeError("image_mask must be a numpy array")
        # check if image_mask is a 2D array
        if image_mask.ndim != 2:
            raise ValueError("image_mask must be a 2D array")
        return True

    @staticmethod
    def check_path(path):
        # check if path is a numpy array
        if not isinstance(path, np.ndarray):
            raise TypeError("path must be a numpy array")
        # check if path is a 2D array
        if path.ndim != 2:
            raise ValueError("path must be a 2D array")
        # check if path has 2 columns
        if path.shape[1] != 2:
            raise ValueError("path must have 2 columns")
        # check if path has at least 2 rows
        if path.shape[0] < 2:
            raise ValueError("path must have at least 2 rows")
        return True

    @staticmethod
    def check_position(position):
        # check if position is a numpy array
        if not isinstance(position, np.ndarray):
            raise TypeError("position must be a numpy array")
        # check if position is a 1D array
        if position.ndim != 1:
            raise ValueError("position must be a 1D array")
        # check if position has 2 elements
        if position.shape[0] != 2:
            raise ValueError("position must have 2 elements")
        return True

    def log_init_image(self, init_image):
        if Reporter.check_rgb_image(init_image):
            self._init_image = init_image
        if self._debug_plot:
            plt.imshow(init_image, origin="lower")
            # set image title
            plt.title("Initial Image")
            # plt.show()
            plt.savefig(f"{self._report_file_base}_init_image.png")

    def log_obstacle_mask(self, obstacle_mask):
        if Reporter.check_image_mask(obstacle_mask):
            self._obstacle_mask = obstacle_mask
        if self._debug_plot:
            plt.imshow(obstacle_mask, cmap="gray", origin="lower")
            # set image title
            plt.title("Mask of Obstacles")
            # plt.show()
            plt.savefig(f"{self._report_file_base}_obstacle_mask.png")

    def log_plan_path(self, plan_path):
        if Reporter.check_path(plan_path):
            self._plan_path = plan_path
        if self._debug_plot:
            plt.imshow(self._init_image, cmap="gray", origin="lower")
            # set image title
            plt.title("Plan Path on Initial Image")
            plt.plot(plan_path[:, 0], plan_path[:, 1], "r-")
            # plt.show()
            plt.savefig(f"{self._report_file_base}_plan_path.png")

    def log_start_position(self, start_position):
        if Reporter.check_position(start_position):
            self._start_position = start_position

    def log_goal_position(self, goal_position):
        if Reporter.check_position(goal_position):
            self._goal_position = goal_position

    def log_robot_sim_state(self, robot_sim_position, robot_orientation, sim_time):
        if Reporter.check_position(robot_sim_position):
            self._robot_sim_path = np.insert(
                self._robot_sim_path, self._robot_sim_path.shape[0], values=robot_sim_position, axis=0
            )
        if not isinstance(robot_orientation, (int, float)):
            raise TypeError("robot_orientation must be a number")
        self._robot_sim_orientation = np.insert(
            self._robot_sim_orientation, self._robot_sim_orientation.shape[0], values=robot_orientation, axis=0
        )
        # log simulation time
        if not isinstance(sim_time, (int, float)):
            raise TypeError("sim_time must be a number")
        if sim_time < 0:
            raise ValueError("sim_time must be a non-negative number")
        # check if sim_time is increasing
        if self._robot_sim_time.size > 0 and sim_time <= self._robot_sim_time[-1]:
            raise ValueError("sim_time must be increasing")
        self._robot_sim_time = np.insert(self._robot_sim_time, self._robot_sim_time.shape[0], values=sim_time, axis=0)
        

    def check_plan_data(self):
        # check if all required data is logged
        if self._init_image is None:
            raise ValueError("init_image is not logged")
        if self._obstacle_mask is None:
            raise ValueError("obstacle_mask is not logged")
        if self._plan_path is None:
            raise ValueError("plan_path is not logged")
        if self._start_position is None:
            raise ValueError("start_position is not logged")
        if self._goal_position is None:
            raise ValueError("goal_position is not logged")
        return True

    def report_plan(self, plot=True, display=True):
        # check if all required data is logged
        if self.check_plan_data():
            self.calc_paln_path_length()
            self.calc_plan_path_to_obstacle_distance()
            self.check_if_plan_path_properly_smoothed()
            self.save()
            if plot:
                self.plot_plan_path()
            if display:
                self.display_report()


    def check_goal_error(self):
        goal_error = np.linalg.norm(self._robot_sim_path[-1] - self._goal_position)
        if goal_error > self.__goal_tolerance:
            self._report_str += "Goal not reached, offset: {:.2f}\n".format(goal_error)
            return 0
        else:
            self._report_str += "Goal reached, offset: {:.2f}\n".format(goal_error)
            return 1

    def check_if_plan_path_properly_smoothed(self):
        # check is the plan path is properly smoothed
        # calculate the angle between two consecutive points in the plan path
        angles = np.empty(shape=(self._plan_path.shape[0] - 1, 1))
        for i in range(1, self._plan_path.shape[0]):
            # calculate the angle between two consecutive points
            h_diff = self._plan_path[i, 0] - self._plan_path[i - 1, 0]
            w_diff = self._plan_path[i, 1] - self._plan_path[i - 1, 1]
            angle = np.arctan2(h_diff, w_diff) * 180 / np.pi
            angles[i - 1] = angle
        # calculate the difference between consecutive angles
        angle_diff = np.empty(shape=(angles.shape[0] - 1, 1))
        for i in range(1, angles.shape[0]):
            diff1 = np.abs(angles[i] - angles[i - 1])
            diff2 = np.abs(angles[i] - angles[i - 1] + 360)
            diff3 = np.abs(angles[i] - angles[i - 1] - 360)
            angle_diff[i - 1] = np.min([diff1, diff2, diff3])
        # calculate the average angle difference, skip the first empty element
        avg_angle_diff = np.mean(angle_diff)
        # add to report string
        self._report_str += "Average path point angle difference: {:.2f}\n".format(avg_angle_diff)
        max_angle_diff = np.max(angle_diff)
        # add to report string
        self._report_str += "Max path point angle difference: {:.2f}\n".format(max_angle_diff)
        return avg_angle_diff

    def calc_point_to_obstacle_distance(self, point):
        # calculate distance from point to nearest obstacle
        point_rc = point[1], point[0]  # convert to row, column format
        distance = np.min(np.linalg.norm(point_rc - np.argwhere(self._obstacle_mask == 0), axis=1))
        return distance

    def calc_point_to_path_distance(self, point):
        # calculate distance from point to nearest point in the plan path
        distance = np.min(np.linalg.norm(point - self._plan_path, axis=1))
        return distance

    def calc_robot_sim_path_deviation_from_plan_path(self):
        # calculate the deviation of the robot sim path from the plan path
        deviations = np.empty(shape=(self._robot_sim_path.shape[0], 1))
        for i, point in enumerate(self._robot_sim_path):
            deviations[i] = self.calc_point_to_path_distance(point)

        average_deviation = np.mean(deviations)
        # add to report string
        self._report_str += "Average robot path deviation from plan path: {:.2f}\n".format(average_deviation)
        aggregate_deviation = np.sum(deviations)
        # add to report string
        self._report_str += "Robot path deviation from plan path: {:.2f}\n".format(np.max(aggregate_deviation))
        return deviations

    def calc_robot_sim_path_time(self):
        # calculate the time taken by the robot sim to reach the goal
        time_taken = self._robot_sim_time[-1] - self._robot_sim_time[0]
        # add to report string
        self._report_str += "Robot path  time: {:.2f} seconds\n".format(time_taken[0])
        return time_taken

    def estimate_robot_sim_energy(self):
        # calculate the moving speed of robot
        moving_speed = np.empty(shape=(self._robot_sim_path.shape[0] - 1, 1))
        for i in range(1, self._robot_sim_path.shape[0]):
            moving_speed[i - 1] = np.linalg.norm(self._robot_sim_path[i] - self._robot_sim_path[i - 1]) / (
                self._robot_sim_time[i] - self._robot_sim_time[i - 1]
            )
        # calculate the rotation speed of robot
        rotation_speed = np.empty(shape=(self._robot_sim_path.shape[0] - 1, 1))
        for i in range(1, self._robot_sim_path.shape[0]):
            # 修正计算方法，将角度差值限制在0到π之间，假定机器人会选择最近的旋转方向
            rotation_diff = np.abs(self._robot_sim_orientation[i] - self._robot_sim_orientation[i - 1])
            if rotation_diff > np.pi:
                rotation_diff = 2 * np.pi - rotation_diff
            rotation_speed[i - 1] = rotation_diff / (
                self._robot_sim_time[i] - self._robot_sim_time[i - 1]
            )

        # calculate the energy of robot
        # calculate the difference between consecutive moving speeds
        moving_speed_diff = np.empty(shape=(moving_speed.shape[0] - 1, 1))
        for i in range(1, moving_speed.shape[0]):
            moving_speed_diff[i - 1] = np.abs(moving_speed[i] - moving_speed[i - 1])
        # calculate the difference between consecutive rotation speeds
        rotation_speed_diff = np.empty(shape=(rotation_speed.shape[0] - 1, 1))
        for i in range(1, rotation_speed.shape[0]):
            rotation_speed_diff[i - 1] = np.abs(rotation_speed[i] - rotation_speed[i - 1])
        # normalize the moving speed and rotation speed
        moving_speed_diff = moving_speed_diff / self._max_speed
        rotation_speed_diff = rotation_speed_diff / self._max_rotation_speed

        # estimate the energy of robot by summing the moving speed and rotation speed
        weight_of_moving_speed = 0.2
        weight_of_rotation_speed = 0.8
        energy = np.sum(weight_of_moving_speed * moving_speed_diff + weight_of_rotation_speed * rotation_speed_diff)
        # add to report string
        self._report_str += "Robot path energy: {:.2f}\n".format(energy)
        return energy

    def calc_sim_path_to_obstacle_distance(self):
        # calculate distance from each point in the robot sim path to the nearest obstacle
        distances = np.empty(shape=(self._robot_sim_path.shape[0], 1))
        for i, point in enumerate(self._robot_sim_path):
            distances[i] = self.calc_point_to_obstacle_distance(point)
        # add to report string
        self._report_str += "Min robot path to obstacle distance: {:.2f}\n".format(np.min(distances))
        # add the average distance to the report string
        self._report_str += "Average robot path to obstacle distance: {:.2f}\n".format(np.mean(distances))
        return distances

    def calc_plan_path_to_obstacle_distance(self):
        # calculate distance from each point in the plan path to the nearest obstacle
        distances = np.empty(shape=(self._plan_path.shape[0], 1))
        for i, point in enumerate(self._plan_path):
            distances[i] = self.calc_point_to_obstacle_distance(point)
        # add to report string
        self._report_str += "Min plan path to obstacle distance: {:.2f}\n".format(np.min(distances))
        # add the average distance to the report string
        self._report_str += "Average plan path to obstacle distance: {:.2f}\n".format(np.mean(distances))
        return distances

    def calc_paln_path_length(self):
        length = np.sum(np.linalg.norm(self._plan_path[1:] - self._plan_path[:-1], axis=1))
        # add to report string
        self._report_str += "Plan path length: {:.2f}\n".format(length)
        return length

    def calc_robot_sim_path_length(self):
        length = np.sum(np.linalg.norm(self._robot_sim_path[1:] - self._robot_sim_path[:-1], axis=1))
        # add to report string
        self._report_str += "Robot path length: {:.2f}\n".format(length)
        return length

    def display_report(self):
        print(self._report_str)

    def save(self):
        with open(self._report_file, "w") as f:
            f.write(self._report_str)

    def inflate_obstacles(self):
        kernel = np.ones((self.__inflate_size, self.__inflate_size), np.uint8)
        inflated_obstacle_mask = cv2.erode(self._obstacle_mask, kernel, iterations=1)
        return inflated_obstacle_mask

    def report_all(self):
        self.report_plan(plot=False, display=False)
        if self._robot_sim_path.size == 0:
            raise ValueError("robot_sim_path is not logged")
        
        self.plot_summary_results()
        
        self.calc_sim_path_to_obstacle_distance()
        self.calc_robot_sim_path_length()
        self.calc_robot_sim_path_deviation_from_plan_path()
        self.calc_robot_sim_path_time()
        self.estimate_robot_sim_energy()
        self.check_goal_error()

        self.save()
        self.display_report()

    def plot_plan_path(self):
        fig, ax = plt.subplots()
        ax.imshow(self.inflate_obstacles(), cmap="gray", alpha=0.5, origin="lower")
        ax.plot(self._plan_path[:, 0], self._plan_path[:, 1], "r-")
        ax.plot(self._start_position[0], self._start_position[1], "bo")
        ax.plot(self._goal_position[0], self._goal_position[1], "go")
        # set image title
        plt.title("Plan Path on Inflated Obstacle Image")
        # plt.show()
        plt.savefig(f"{self._report_file_base}_plan_path.png")

    def _calculate_velocities_from_path(self):
        """从记录的路径数据计算线速度和角速度"""
        if self._robot_sim_path.shape[0] < 2:
            return np.array([0]), np.array([0])

        # 计算时间差
        dt = np.diff(self._robot_sim_time.flatten())
        # 避免除以零
        dt[dt == 0] = 1e-6

        # 计算线速度 (m/s)
        path_diff = np.diff(self._robot_sim_path, axis=0)
        distances_px = np.linalg.norm(path_diff, axis=1)
        linear_velocities_px_s = distances_px / dt
        linear_velocities_m_s = linear_velocities_px_s * self.pixels_to_meters_scale
        
        # 计算角速度 (rad/s)
        orient_diff = np.diff(self._robot_sim_orientation.flatten())
        # 标准化角度差到 [-pi, pi] 以获得最短旋转路径
        orient_diff = (orient_diff + np.pi) % (2 * np.pi) - np.pi
        angular_velocities = orient_diff / dt

        # 补全第一个时间步的速度为0，以匹配时间数组长度
        velocities = np.insert(linear_velocities_m_s, 0, 0)
        omegas = np.insert(angular_velocities, 0, 0)

        return velocities, omegas
    
    def log_planning_time(self, planning_time):
        """
        记录路径规划所用的时间。
        :param planning_time: 路径规划所用的时间，单位为秒。
        """
        if not isinstance(planning_time, (int, float)):
            raise TypeError("planning_time must be a number")
        if planning_time < 0:
            raise ValueError("planning_time must be a non-negative number")
        
        self._report_str += f"路径规划时间: {planning_time:.2f} 秒\n"
    
    def plot_summary_results(self):
        """
        绘制一个包含轨迹、速度、角速度和路径误差的四宫格总结图。
        """
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("机器人路径跟踪性能总结", fontsize=16)

        velocities, omegas = self._calculate_velocities_from_path()
        
        # --- 1. 轨迹图 ---
        ax1 = axs[0, 0]
        # ax1.imshow(self.inflate_obstacles(), cmap="gray", alpha=0.5, origin="lower")
        ax1.imshow(self._init_image, origin="lower")
        ax1.plot(self._plan_path[:, 0], self._plan_path[:, 1], "r-", label="规划路径")
        ax1.plot(self._robot_sim_path[:, 0], self._robot_sim_path[:, 1], "b-", label="实际轨迹")
        ax1.plot(self._start_position[0], self._start_position[1], "bo", markersize=8, label="起点")
        ax1.plot(self._goal_position[0], self._goal_position[1], "go", markersize=8, label="终点")
        ax1.set_title("路径跟踪轨迹")
        ax1.set_xlabel("X [像素]")
        ax1.set_ylabel("Y [像素]")
        ax1.legend()
        ax1.axis('equal')
        ax1.grid(True)

        # --- 2. 线速度图 ---
        ax2 = axs[0, 1]
        ax2.plot(self._robot_sim_time, velocities, "r-", label="线速度 (v)")
        ax2.set_title("线速度变化")
        ax2.set_xlabel("时间 [s]")
        ax2.set_ylabel("速度 [m/s]")
        ax2.grid(True)
        ax2.legend()

        # --- 3. 角速度图 ---
        ax3 = axs[1, 0]
        ax3.plot(self._robot_sim_time, omegas, "b-", label="角速度 (ω)")
        ax3.set_title("角速度变化")
        ax3.set_xlabel("时间 [s]")
        ax3.set_ylabel("角速度 [rad/s]")
        ax3.grid(True)
        ax3.legend()

        # --- 4. 路径跟踪误差图 ---
        ax4 = axs[1, 1]
        deviations = self.calc_robot_sim_path_deviation_from_plan_path()
        ax4.plot(self._robot_sim_time, deviations, "g-", label="路径跟踪误差")
        ax4.set_title("路径跟踪误差")
        ax4.set_xlabel("时间 [s]")
        ax4.set_ylabel("误差 [像素]")
        ax4.grid(True)
        ax4.legend()

        # 调整布局并保存/显示
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 保存图像
        plot_filename = f"{self._report_file_base}.png"
        plt.savefig(plot_filename)
        print(f"总结图已保存至: {plot_filename}")

        if self._debug_plot:
            plt.show()
        
