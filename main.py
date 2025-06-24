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
import math
import argparse

from reporter import Reporter
from trackers import *
from planner import *
from vehicle_model import *
from robot import Robot


def get_pixel_to_meter_scale(camera_height, fov_degrees, resolution):
    # 从仿真环境中计算像素到米的转换比例
    # 计算地面视场宽度
    fov_radians = math.radians(fov_degrees)
    ground_half_width = camera_height * math.tan(fov_radians / 2)
    ground_width = 2 * ground_half_width
    
    # 像素到米的转换比例
    pixels_to_meters_scale = ground_width / resolution
    return pixels_to_meters_scale


pixels_to_meters_scale = get_pixel_to_meter_scale(camera_height=5.0, fov_degrees=60.0, resolution=512)
sim_step = 0.05  # 仿真步长为 50ms（与 VREP 设置一致）


planners = {
    'AStar': {
        'class': AStarPlanner,
        'params': {'smooth_path': True, 'weight_data': 0.001, 'weight_smooth': 0.8}
    },
    'RRT': {
        'class': RRTPlanner,
        'params': {'expand_dis': 5, 'path_resolution': 1, 'goal_sample_rate': 20, 'max_iter': 5000, 'smooth_path': True, 'weight_data': 0.01, 'weight_smooth': 0.9}
    },
    'RRTStar': {
        'class': RRTStarPlanner,
        'params': {'expand_dis': 5, 'path_resolution': 1, 'goal_sample_rate': 20, 'max_iter': 5000, 'smooth_path': True, 'weight_data': 0.01, 'weight_smooth': 0.9, 'connect_circle_dist': 50.0}
    }
}

controllers = {
    'DifferentialDrive': {
        'PurePursuit': {'class': PurePursuitAngularController, 'params': {'lookahead_distance': 30}},
        'Stanley': {'class': StanleyAngularController, 'params': {'WB': 0.25, 'k_gain': 5}},
        'MPC': {
            'class': DifferentialDriveMPCAngularController,
            'params': {'state_weight': np.diag([2, 2, 0.25, 0]), 'control_weight': np.diag([0, 0.005]), 'terminal_weight': np.diag([4, 4, 1, 0]), 'control_change_weight': np.diag([0, 0.01]), 'horizon_length': 10, 'dt': sim_step, 'control_speed': False}
        },
        'MPC_Speed': {
            'class': DifferentialDriveMPCAngularController,
            'params': {'state_weight': np.diag([2, 2, 0.6, 0.25]), 'control_weight': np.diag([0.1, 0.015]), 'terminal_weight': np.diag([4, 4, 1, 1]), 'control_change_weight': np.diag([0.01, 0.05]), 'control_speed': True, 'MAX_ACCELERATION': 5, 'horizon_length': 10, 'dt': sim_step, 'MAX_V': 1.3, 'MAX_OMEGA': 3.0}
        },
        'PI': {'class': PISpeedController, 'params': {'kp': 5, 'ki': 0.1, 'n': 5, 'max_acc': 5, 'min_acc': -5}}
    },
    'AckermannSteering': {
        'PurePursuit': {'class': PurePursuitAngularController, 'params': {'lookahead_distance': 30}},
        'Stanley': {'class': StanleyAngularController, 'params': {'k_gain': 5}},
        'MPC': {
            'class': DifferentialDriveMPCAngularController,
            'params': {'state_weight': np.diag([2, 2, 0.25, 0]), 'control_weight': np.diag([0, 0.005]), 'terminal_weight': np.diag([4, 4, 1, 0]), 'control_change_weight': np.diag([0, 0.01]), 'horizon_length': 10, 'dt': sim_step, 'control_speed': False}
        },
        'MPC_Speed': {
            'class': DifferentialDriveMPCAngularController,
            'params': {'state_weight': np.diag([2, 2, 0.5, 0.5]), 'control_weight': np.diag([0.005, 0.005]), 'terminal_weight': np.diag([4, 4, 1, 1]), 'control_change_weight': np.diag([0.01, 0.01]), 'control_speed': True, 'MAX_ACCELERATION': 5, 'horizon_length': 10, 'dt': sim_step, 'MAX_V': 1.5}
        },
        'PI': {'class': PISpeedController, 'params': {'kp': 3, 'ki': 0.05, 'n': 5, 'max_acc': 3, 'min_acc': -3}}
    }
}

def get_robot(args, clientID):
    # --- 1. 选择并实例化规划器 ---
    planner_config = planners.get(args.planner)
    if not planner_config:
        raise ValueError(f"Unsupported planner: {args.planner}")
    planner = planner_config['class'](**planner_config['params'])
    
    # --- 2. 选择并实例化控制器 ---
    vehicle_controllers_config = controllers.get(args.vehicle_model)
    if not vehicle_controllers_config:
        raise ValueError(f"Unsupported vehicle model: {args.vehicle_model}")

    # 收集控制器参数用于报告
    config_details = {'planner': planner_config['params']}

    if args.speed == 'MPC':
        if args.angular != 'MPC':
            raise ValueError("When using MPC for speed control, angular controller must also be MPC.")
        angular_config = vehicle_controllers_config.get('MPC_Speed')
        angular_controller = angular_config['class'](pixels_to_meters_scale=pixels_to_meters_scale, **angular_config['params'])
        tracker = PathTracker(angular_controller=angular_controller)
        config_details['angular_controller'] = angular_config['params']
    else:
        angular_config = vehicle_controllers_config.get(args.angular)
        speed_config = vehicle_controllers_config.get(args.speed)
        if not angular_config or not speed_config:
            raise ValueError(f"Unsupported controller combination for {args.vehicle_model}: {args.angular}/{args.speed}")
        
        angular_controller = angular_config['class'](pixels_to_meters_scale=pixels_to_meters_scale, **angular_config['params'])
        speed_controller = speed_config['class'](**speed_config['params'])
        tracker = PathTracker(angular_controller=angular_controller, speed_controller=speed_controller)
        config_details['angular_controller'] = angular_config['params']
        config_details['speed_controller'] = speed_config['params']
        

    if args.vehicle_model == 'DifferentialDrive':
        params = VehicleParameters(
            wheel_radius=0.04, wheelbase=0.2,
            robot_radius_px=20,
        )
        
        _, leftMotor = sim.simxGetObjectHandle(clientID, "bubbleRob_leftMotor", sim.simx_opmode_blocking)
        _, rightMotor = sim.simxGetObjectHandle(clientID, "bubbleRob_rightMotor", sim.simx_opmode_blocking)
        sim.simxSetJointTargetVelocity(clientID, leftMotor, 0, sim.simx_opmode_oneshot)
        sim.simxSetJointTargetVelocity(clientID, rightMotor, 0, sim.simx_opmode_oneshot)
        
        vehicle_model_instance = DifferentialDriveModel(params=params, left_motor=leftMotor, right_motor=rightMotor)
    elif args.vehicle_model == 'AckermannSteering':
        params = VehicleParameters(
            wheel_radius=0.04, wheelbase=0.12,
            track_width=0.12, max_steering_angle=math.pi/4,
            robot_radius_px=20,
        )
            
        _, rear_motor = sim.simxGetObjectHandle(clientID, "Ackermann_RearMotor", sim.simx_opmode_blocking)
        _, front_left_steer = sim.simxGetObjectHandle(clientID, "Ackermann_LeftSteer", sim.simx_opmode_blocking)
        _, front_right_steer = sim.simxGetObjectHandle(clientID, "Ackermann_RightSteer", sim.simx_opmode_blocking)
        sim.simxSetJointTargetVelocity(clientID, rear_motor, 0, sim.simx_opmode_oneshot)
        sim.simxSetJointPosition(clientID, front_left_steer, 0, sim.simx_opmode_oneshot)
        sim.simxSetJointPosition(clientID, front_right_steer, 0, sim.simx_opmode_oneshot)
        
        vehicle_model_instance = AckermannSteeringModel(
            params=params, rear_motor=rear_motor,
            front_left_steer=front_left_steer, front_right_steer=front_right_steer,
            clientID=clientID,
        )
    else:
        raise ValueError(f"Unsupported vehicle model: {args.vehicle_model}")
    
    config_details['vehicle_model'] = {
        'type': args.vehicle_model,
        'params': params.__dict__
    }
    target_speed = 1
    config_details['target_speed'] = {'target_speed': target_speed}
    

    robot = Robot(
        clientID,
        planner=planner,
        tracker=tracker,
        vehicle_model=vehicle_model_instance,
        target_speed=target_speed,
    )
    
    return robot, config_details

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="CoppeliaSim Remote API Example")
    args.add_argument('--planner', type=str, default='AStar', choices=['AStar', 'RRT', 'RRTStar'],
                    help='选择路径规划器: AStar, RRT, RRTStar')
    args.add_argument('--angular', type=str, default='PurePursuit', choices=['PurePursuit', 'Stanley', 'MPC'],
                    help='选择路径跟踪器: PurePursuit, Stanley, MPC')
    args.add_argument('--speed', type=str, default='PI', choices=['PI', 'MPC'],
                    help='选择速度控制器: PI, MPC')
    args.add_argument('--vehicle_model', type=str, default='DifferentialDrive', choices=['DifferentialDrive', 'AckermannSteering'],)
    args = args.parse_args()
    
    print ('程序开始运行')
    sim.simxFinish(-1)
    clientID=sim.simxStart('127.0.0.1',19997,True,True,5000,5)
    if clientID!=-1:
        print ('成功连接到远程API服务器')
        # enable the synchronous mode on the client:
        sim.simxSynchronous(clientID, True)

        # start the simulation:
        sim.simxStartSimulation(clientID,sim.simx_opmode_blocking)
        
        print("初始化规划器、跟踪器以及机器人实例")
        robot, config_details = get_robot(args, clientID)
        reporter = Reporter(args=args, config_details=config_details, pixels_to_meters_scale=pixels_to_meters_scale)
        
        # 确保传感器数据流初始化
        for _ in range(5):
            ret, resolution, image_raw = sim.simxGetVisionSensorImage(robot.clientID,robot.perspectiveSensor,0,sim.simx_opmode_streaming)
            sim.simxSynchronousTrigger(clientID)
            sim.simxGetPingTime(clientID)
            if ret == sim.simx_return_ok and robot.get_sensorImage_info() is not None:
                print("传感器数据流初始化成功")
                break
        
        # 路径规划循环
        retry_num = 0
        print("开始路径规划...")
        while not robot.valid_plan and retry_num < 5:
            print("尝试进行路径规划...")
            robot.init_plan(reporter)
            sim.simxSynchronousTrigger(clientID)
            sim.simxGetPingTime(clientID)
            if not robot.valid_plan:
                print("路径规划失败，重新尝试...")
            else:
                print("路径规划成功！")
            retry_num += 1
        if not robot.valid_plan:
            print("路径规划失败，无法继续仿真。")
            sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)
            sim.simxGetPingTime(clientID)
            sim.simxFinish(clientID)
            exit(0)
                
        # 同步运行控制循环
        print("开始控制循环...")
        while not robot.path_completed:
            robot.sysCall_actuation(reporter)
            sim.simxSynchronousTrigger(clientID)
            sim.simxGetPingTime(clientID)
        
        # 到达终点后，调用report_all输出结果
        reporter.report_all()

        # 停止传感器流
        robot.discontinue_stream()
        
        # stop the simulation:
        sim.simxStopSimulation(clientID,sim.simx_opmode_blocking)
        sim.simxGetPingTime(clientID)
        # Now close the connection to CoppeliaSim:
        sim.simxFinish(clientID)
            
    else:
        print ('未能连接到远程API服务器，请检查CoppeliaSim是否运行并且远程API插件已加载。')
    print ('程序结束运行')