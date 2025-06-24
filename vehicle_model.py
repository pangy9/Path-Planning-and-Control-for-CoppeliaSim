from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
import numpy as np

import sim
from trackers import MotionCommand
from utils import PIControl, angle_mod

@dataclass
class VehicleState:
    """车辆状态"""
    position: tuple        # (x, y) 位置
    orientation: float     # 朝向角 rad
    v: float = 0.0
    omega: float = 0.0
    
@dataclass
class VehicleParameters:
    """车辆参数"""
    wheelbase: float           # 轴距 m
    wheel_radius: float        # 轮半径 m  
    # 阿克曼转向车专用参数
    track_width: float = None   # 左右轮的轮距 m
    max_steering_angle: float = None  # 最大转向角 rad
    robot_radius_px: int = 30 # 车辆半径像素 (用于碰撞检测等)

class VehicleModel(ABC):
    """车辆模型抽象基类"""
    
    def __init__(self, params: VehicleParameters, **args):
        self.params = params
        self.state = VehicleState((0, 0), 0)
    
    @abstractmethod
    def translate_motion_command_to_control_command(self, command: MotionCommand) -> dict:
        """
        执行运动指令，返回具体的控制量
        Returns:
            dict: 具体控制量，如 {"left_wheel_speed": x, "right_wheel_speed": y}
                 或 {"steering_angle": x, "throttle": y}
        """
        pass
    
    @abstractmethod
    def update_state(self, **kargs):
        """更新车辆状态"""
        pass
    

class DifferentialDriveModel(VehicleModel):
    """差速驱动车辆模型"""
    
    def __init__(self, params: VehicleParameters, left_motor, right_motor, **kwargs):
        super().__init__(params)
        self.left_motor = left_motor
        self.right_motor = right_motor
        self.return_steering = False
        
    def translate_motion_command_to_control_command(self, command: MotionCommand) -> dict:
        """将运动指令转换为左右轮速度"""
        v = command.v
        omega = command.omega
        
        
        # 差速驱动运动学
        L = self.params.wheelbase
        R = self.params.wheel_radius
        
        left_speed = v - omega * L / 2
        right_speed = v + omega * L / 2
        
        return {
            "left_wheel_speed": left_speed / R,
            "right_wheel_speed": right_speed / R,
            "v": v,
            "omega": omega
        }
    
    def get_control_handles(self, control_commands: dict) -> list:
        """获取控制手柄字典"""
        return [
            {"handle": self.left_motor, "value": control_commands["left_wheel_speed"], "type": "speed"},
            {"handle": self.right_motor, "value": control_commands["right_wheel_speed"], "type": "speed"}
        ]
    
    def update_state(self, position=None, orientation=None, v=None, omega=None, **kargs):
        """选择性部分更新车辆状态"""
        if position is not None:
            self.state.position = position
        if orientation is not None:
            self.state.orientation = orientation
        if v is not None:
            self.state.v = v
        if omega is not None:
            self.state.omega = omega
    

class AckermannSteeringModel(VehicleModel):
    """阿克曼转向车辆模型 - 前轮转向，后轮统一驱动"""
    
    def __init__(self, params: VehicleParameters, rear_motor, 
                 front_left_steer, front_right_steer, clientID,**kwargs):
        """
        初始化阿克曼转向车模型
        
        Args:
            params: 车辆参数
            rear_motor: 后轮驱动电机句柄（驱动两个后轮）
            front_left_steer: 前左轮转向关节句柄
            front_right_steer: 前右轮转向关节句柄
        """
        super().__init__(params)
        self.return_steering = True
        
        # 电机和转向关节句柄
        self.rear_motor = rear_motor           # 后轮统一驱动电机
        self.front_left_steer = front_left_steer   # 前左轮转向关节
        self.front_right_steer = front_right_steer # 前右轮转向关节
        self.clientID = clientID
        self.left_steering_control = PIControl(8, 2, 4, max_signal=10, min_signal=-10)
        self.right_steering_control = PIControl(8, 2, 4, max_signal=10, min_signal=-10)
        
        # 阿克曼转向相关参数
        self.track_width = getattr(params, 'track_width', 1.5)
        self.max_steering_angle = getattr(params, 'max_steering_angle', math.pi/4)
        sim.simxGetJointPosition(self.clientID, self.front_left_steer, sim.simx_opmode_streaming)
        sim.simxGetJointPosition(self.clientID, self.front_right_steer, sim.simx_opmode_streaming)        
        # 当前转向角
        self.current_steering_angle = 0.0
        
    def translate_motion_command_to_control_command(self, command: MotionCommand) -> dict:
        """将运动指令转换为阿克曼转向控制指令"""
        v = command.v
        omega = command.omega
        
        # 根据命令类型处理
        if command.command_type == "steering":
            # 直接使用转向角
            center_steering_angle = command.steering_angle
        else:
            # 从角速度计算中心转向角
            if abs(v) > 0.01:  # 避免除零
                if v > 0:  # 前进
                    center_steering_angle = math.atan(omega * self.params.wheelbase / v)
                else:  # 倒车 (v < 0)
                    # 倒车时，相同的omega应该产生相反的转向角
                    center_steering_angle = -math.atan(omega * self.params.wheelbase / abs(v))
            else:
                center_steering_angle = 0.0
        
        # 限制转向角范围
        center_steering_angle = np.clip(center_steering_angle, 
                                    -self.max_steering_angle, 
                                    self.max_steering_angle)
        
        # 计算左右前轮的实际转向角（阿克曼几何）
        left_steer_angle, right_steer_angle = self._calculate_wheel_steering_angles(center_steering_angle)
        
        actual_left_steer_angle = sim.simxGetJointPosition(self.clientID, self.front_left_steer, sim.simx_opmode_buffer)[1]
        actual_right_steer_angle = sim.simxGetJointPosition(self.clientID, self.front_right_steer, sim.simx_opmode_buffer)[1]
        left_steer_target_omega = self.left_steering_control.control(angle_mod(left_steer_angle - actual_left_steer_angle))
        right_steer_target_omega = self.right_steering_control.control(angle_mod(right_steer_angle - actual_right_steer_angle))
        
        # 计算后轮驱动速度（可以为负值实现倒车）
        rear_wheel_speed = v / self.params.wheel_radius
        
        return {
            "rear_motor_speed": rear_wheel_speed,
            "front_left_steer_angle": left_steer_angle,
            "front_right_steer_angle": right_steer_angle,
            "left_steer_target_omega": left_steer_target_omega,
            "right_steer_target_omega": right_steer_target_omega,
            "center_steering_angle": center_steering_angle,
            "v": v,
            "omega": omega
        }
    
    def _calculate_wheel_steering_angles(self, center_steering_angle):
        """
        根据中心转向角(自行车模型)计算左右前轮的实际转向角
        
        阿克曼转向几何：内侧轮转角大于外侧轮转角
        """
        if abs(center_steering_angle) < 1e-6:
            # 直行时，两个前轮转向角都为0
            return 0.0, 0.0
        
        L = self.params.wheelbase  # 轴距
        W = self.track_width       # 轮距
        
        # 计算转弯半径（到后轴中心）
        if abs(math.tan(center_steering_angle)) < 1e-6:
            # 防止除零错误
            return center_steering_angle, center_steering_angle
        
        R = L / math.tan(abs(center_steering_angle))
        
        if center_steering_angle > 0:  # 左转
            # 左转时：左轮是内侧轮，右轮是外侧轮
            R_left = R - W/2   # 左轮转弯半径（内侧，更小）
            R_right = R + W/2  # 右轮转弯半径（外侧，更大）
            
            left_angle = math.atan(L / R_left)   # 内侧轮转角更大
            right_angle = math.atan(L / R_right) # 外侧轮转角更小
            
        else:  # 右转
            # 右转时：右轮是内侧轮，左轮是外侧轮
            R_left = R + W/2   # 左轮转弯半径（外侧，更大）
            R_right = R - W/2  # 右轮转弯半径（内侧，更小）
            
            left_angle = -math.atan(L / R_left)   # 外侧轮转角更小
            right_angle = -math.atan(L / R_right) # 内侧轮转角更大
        
        return left_angle, right_angle
    
    def get_control_handles(self, control_commands: dict) -> list:
        """获取控制手柄字典"""
        return [
            {"handle": self.front_left_steer, "value": control_commands["left_steer_target_omega"], "type": "speed"},
            {"handle": self.front_right_steer, "value": control_commands["right_steer_target_omega"], "type": "speed"},
            {"handle": self.rear_motor, "value": control_commands["rear_motor_speed"], "type": "speed"},
        ]
    
    def update_state(self, position=None, orientation=None, v=None, omega=None, 
                     steering_angle=None, **kwargs):
        """更新车辆状态"""
        
        if position is not None:
            self.state.position = position
        if orientation is not None:
            self.state.orientation = orientation
        if v is not None:
            self.state.v = v
        if omega is not None:
            self.state.omega = omega
        if steering_angle is not None:
            self.current_steering_angle = steering_angle
    
