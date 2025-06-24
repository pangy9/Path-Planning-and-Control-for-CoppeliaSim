# This small example illustrates how to use the remote API
# synchronous mode. The synchronous mode needs to be
# pre-enabled on the server side. You would do this by
# starting the server (e.g. in a child script) with:
#
# simRemoteApi.start(19999,1300,false,true)
#
# But in this example we try to connect on port
# 19997 where there should be a continuous remote API
# server service already running and pre-enabled for
# synchronous mode.
#
#
# IMPORTANT: for each successful call to simxStart, there
# should be a corresponding call to simxFinish at the !
try:
    import sim
except:
    print ('--------------------------------------------------------------')
    print ('"sim.py" could not be imported. This means very probably that')
    print ('either "sim.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "sim.py"')
    print ('--------------------------------------------------------------')
    print ('')

import numpy as np
import cv2
import math
from time import time

from trackers import MotionCommand


"""
统一使用数学坐标系：左下角原点，Y轴向上
"""

class Robot():
    def __init__(
        self,
        clientID,
        planner,
        tracker,
        vehicle_model,
        target_speed=1.0,
        display_tracking=True,
    ):
        self.clientID = clientID
        
        # 策略注入
        self.vehicle_model = vehicle_model
        self.planner = planner
        self.tracker = tracker
        
        self.goal_reached_threshold = 10 # 目标点到达阈值，单位像素
        self.target_speed = target_speed # 目标速度，单位 m/s
        self.robot_radius_px = vehicle_model.params.robot_radius_px
        self.display_tracking = display_tracking
        self.return_steering = vehicle_model.return_steering
        
        # 初始化传感器流
        _, self.perspectiveSensor=sim.simxGetObjectHandle(clientID,"top_view_camera", sim.simx_opmode_blocking)
        sim.simxGetVisionSensorImage(clientID, self.perspectiveSensor, options=0, operationMode=sim.simx_opmode_streaming)
        
        # 路径规划相关
        self.valid_plan = False
        self.plan_path_xy = []
        self.path_completed = False
        self.last_goal = (0, 0)
            
    def get_sensorImage_info(self):
        '''
        读取视觉传感器信息，返回图像和机器人坐标以及绿盘坐标
        使用数学坐标系: 左下角原点, Y轴向上为正
        return display_image, (center_x, center_y, oritation), (cX_g, cY_g)
        '''
        # 蓝色盘子(r,g,b) = (23, 23, 241)
        # 红色盘子(r,g,b) = (241, 23, 23)
        # 绿色目标(r,g,b) = (23, 241, 23)
        # 障碍物(r,g,b) = (230, 230, 230)
        
        # 读取传感器数据
        ret, resolution, image_raw = sim.simxGetVisionSensorImage(self.clientID,self.perspectiveSensor, 0, sim.simx_opmode_buffer) # Read the perspective sensor
        if ret != sim.simx_return_ok:
            return None
        image = np.array(image_raw, dtype=np.int16) + 256
        image = np.array(image, dtype=np.uint8)
        # 摄像头分辨率分别指代 x 和 y 轴的像素数
        image.resize([resolution[0], resolution[1], 3])
        
        # 原摄像头位置是在仿真环境的正上方，以一个俯视的观察二维视角为基准看，z轴（蓝色）朝地面，x轴（红色）向左，y轴（绿色）向上方。
        # 摄像头方向的 y 轴正方向通常对应 resize 后图像矩阵的行增大方向。x轴方向对应与图像矩阵的列减小方向。
        # 但是plt.imshow从图像行出发，默认模式是左上角原点，y轴向下，即向下时图像矩阵的行增加方向，也就是与仿真环境的 y 轴正方向相反。
        # 为了能够让plt.imshow显示的坐标系与数学坐标系一致，我们需要使用origin='lower'
        display_image = image.copy()

        
        mask_blue = cv2.inRange(image, (23, 23, 241), (25, 25, 255))
        mask_red = cv2.inRange(image, (241, 23, 23), (255, 25, 25))
        mask_green = cv2.inRange(image, (23, 241, 23), (24, 242, 24))

        contours_info_blue = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_info_red = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_info_green = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 获取轮廓列表
        # 应该用opencv-python 3.4.9.33
        if len(contours_info_blue) == 2:  # OpenCV 4.x
            contours_blue = contours_info_blue[0]
            contours_red = contours_info_red[0]
            contours_green = contours_info_green[0]
        else:  # OpenCV 3.x
            contours_blue = contours_info_blue[1]
            contours_red = contours_info_red[1]
            contours_green = contours_info_green[1]
        
        # 计算蓝色、红色圆盘中心来得到机器人位置
        if contours_blue and contours_red:
            cX_b, cY_b, cX_r, cY_r = None, None, None, None

            if contours_blue:
                M_b = cv2.moments(contours_blue[0])
                if M_b["m00"] != 0:
                    cX_b = int(M_b["m10"] / M_b["m00"])
                    cY_b = int(M_b["m01"] / M_b["m00"])

            if contours_red:
                M_r = cv2.moments(contours_red[0])
                if M_r["m00"] != 0:
                    cX_r = int(M_r["m10"] / M_r["m00"])
                    cY_r = int(M_r["m01"] / M_r["m00"])
                    
            center_x = int((cX_b + cX_r) / 2)
            center_y = int((cY_b + cY_r) / 2)
            
            dx = cX_r - cX_b
            dy = cY_r - cY_b
            oritation = math.atan2(dy, dx)
        else:
            print("警告: 蓝色或红色圆盘未找到，无法计算机器人位置和朝向。")
            return None

        # 计算目标位置
        cX_g, cY_g = 1, 1
        if contours_green:
            M_g = cv2.moments(contours_green[0])
            if M_g["m00"] != 0:
                cX_g = int(M_g["m10"] / M_g["m00"])
                cY_g = int(M_g["m01"] / M_g["m00"])

                self.last_goal = (cX_g, cY_g)
        else:
            print("警告: 绿色目标圆盘未找到，使用上次的目标位置。")
            cX_g, cY_g = self.last_goal
        
        return display_image, (center_x, center_y, oritation), (cX_g, cY_g)
    
    def generate_costmap_and_obstacle_map(self, image, robot_radius_px=30):
        '''
        Return costmap, obstacle_map
        '''
        # 获取障碍物掩码
        mask_obstacle = cv2.inRange(image, (230, 230, 230), (232, 232, 232))
        obstacle_map = cv2.bitwise_not(mask_obstacle) # 0是障碍物，255是非障碍物

        # 获取平面外的区域（黑色区域）
        mask_null = cv2.inRange(image, (0, 0, 0), (0, 0, 0))
        null_map = cv2.bitwise_not(mask_null) # 0是黑色区域，255是非黑色区域
        
        resolution = image.shape[:2] # (height, width)
        costmap = np.ones(resolution, dtype=np.float32)
        
        # 创建实际障碍物的掩码 (obstacle_map 中 0 代表障碍物)
        actual_obstacle_mask = (obstacle_map == 0)
        
        # 关键：考虑机器人尺寸进行障碍物膨胀
        kernel_size1 = int(robot_radius_px * 2.8)
        kernel_size2 = int(robot_radius_px * 3.3)
        
        # 创建椭圆形膨胀核
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size1, kernel_size2))
        
        # 膨胀操作：将障碍物边界向外扩展
        dilated_obstacle_zone_mask = cv2.dilate(actual_obstacle_mask.astype(np.uint8), kernel, iterations=1)
        
        costmap[dilated_obstacle_zone_mask == 1] = np.inf
        costmap[obstacle_map == 0] = np.inf
        costmap[null_map == 0] = np.inf
        
        return costmap, obstacle_map
            
    def init_plan(self, reporter=None):
        '''
        根据当前的影像，生成规划
        现在使用数学坐标系：左下角原点，Y轴向上
        '''
        _info = self.get_sensorImage_info()
        if _info is None:
            print("Failed to get sensor image info for planning.")
            return 
        display_image, (center_x, center_y, orientation), (cX_g, cY_g) = _info
        self.vehicle_model.update_state(
            position=(center_x, center_y), 
            orientation=orientation,
        )

        print(f"数学坐标系下 - Start: ({center_x:.1f}, {center_y:.1f}), Goal: ({cX_g:.1f}, {cY_g:.1f})")
        
        costmap, obstacle_map = self.generate_costmap_and_obstacle_map(display_image, robot_radius_px=self.robot_radius_px)

        # 进行路径规划
        start_time = time()
        plan_xy = self.planner.plan(
            start_xy=(center_x, center_y), 
            goal_xy=(cX_g, cY_g), 
            costmap=costmap
        )
        planning_time = time() - start_time
        print(f"路径规划耗时: {planning_time:.4f} 秒")
        
        if plan_xy:
            self.valid_plan = True
            self.plan_path_xy = plan_xy
            
            # 报告
            if reporter:
                reporter.log_planning_time(planning_time)
                reporter.log_init_image(display_image)
                reporter.log_start_position(np.array((center_x, center_y)))
                reporter.log_goal_position(np.array((cX_g, cY_g)))
                reporter.log_obstacle_mask(obstacle_map)
                reporter.log_plan_path(np.array(plan_xy))
            
    def sysCall_actuation(self, reporter=None):
        _info = self.get_sensorImage_info()
        if _info is None:
            return
        display_image, (center_x, center_y, orientation_rad), (_, _) = _info
        
        self.vehicle_model.update_state(
            position=(center_x, center_y), 
            orientation=orientation_rad,
        )

        # 判断是否到达终点 (路径的最后一个点)
        if self.plan_path_xy and len(self.plan_path_xy) > 0:
            is_reached, distance = self._check_goal_reached(self.vehicle_model.state.position, self.plan_path_xy[-1])
            if is_reached:
                print("到达目标点，停止车辆。")
                self.path_completed = True
                self.stop_vehicle()
                if reporter:
                    currentSimTime = sim.simxGetLastCmdTime(self.clientID) / 1000.0
                    reporter.log_robot_sim_state(np.array((center_x, center_y)), orientation_rad, currentSimTime)
                return
        else:
            print("警告: 没有有效路径，暂停小车。")
            self.path_completed = True
            self.stop_vehicle()
            return

        current_pose_px = (center_x, center_y, orientation_rad)
        motion_command = self.tracker.compute_motion_command(
            current_pose_px=current_pose_px,
            path_xy=self.plan_path_xy,
            target_speed=self.target_speed if distance > 100 else 0.5, # 如果距离小于100像素，则减速到合理速度
            display_image=display_image if self.display_tracking else None,
            current_speed=self.vehicle_model.state.v,
            return_steering=self.return_steering,
            WB=self.vehicle_model.params.wheelbase,
        )
        self.set_motion_command(motion_command)
        
        if reporter:
            currentSimTime = sim.simxGetLastCmdTime(self.clientID) / 1000.0
            reporter.log_robot_sim_state(np.array((center_x, center_y)), orientation_rad, currentSimTime)

    def set_motion_command(self, motion_command):
        '''
        设置车辆运动指令
        '''
        control_command = self.vehicle_model.translate_motion_command_to_control_command(motion_command)
        self._set_control_command(control_command)
        self.vehicle_model.update_state(
            v=motion_command.v,
            omega=motion_command.omega,
        )
    
    def stop_vehicle(self):
        '''
        停止车辆
        '''
        motion_command = MotionCommand(
            v=0.0,
            omega=0.0,
            command_type='stop'
        )
        control_command = self.vehicle_model.translate_motion_command_to_control_command(motion_command)
        self._set_control_command(control_command)
        
    def _set_control_command(self, control_command):
        '''
        设置车辆控制命令
        '''
        control_list = self.vehicle_model.get_control_handles(control_command)
        for control in control_list:
            if control['type'] == 'speed':
                sim.simxSetJointTargetVelocity(self.clientID, control['handle'], control['value'], sim.simx_opmode_oneshot)
            else:
                raise ValueError(f"Unsupported control type: {control['type']}")
        
    def _check_goal_reached(self, pos_xy, goal_xy):
        '''
        检查是否到达目标点
        返回 True 表示到达目标点，False 表示未到达
        '''
        dist = np.linalg.norm(np.array(pos_xy) - np.array(goal_xy))
        return dist < self.goal_reached_threshold, dist
    
    def discontinue_stream(self):
        sim.simxGetVisionSensorImage(self.clientID, self.perspectiveSensor, 0, sim.simx_opmode_discontinue)
    
