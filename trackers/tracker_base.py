from dataclasses import dataclass


@dataclass
class MotionCommand:
    """运动指令数据结构"""
    v: float = None    # 线速度 m/s
    omega: float = None   # 角速度 rad/s
    steering_angle: float = None     # 转向角 rad (阿克曼模型用)
    command_type: str = "velocity"  # "velocity" 或 "steering" 或 "stop"


class PathTracker:
    """路径跟踪器 - 组合速度控制器和角度控制器"""
    def __init__(self, angular_controller, speed_controller=None):
        self.angular_controller = angular_controller
        self.speed_controller = speed_controller
        
        # 检查是否是MPC且控制速度
        self.mpc_controls_speed = (
            hasattr(angular_controller, 'control_speed') and 
            angular_controller.control_speed
        )

    def compute_motion_command(self, current_pose_px, path_xy, target_speed=1.0, current_speed=None,
                             return_steering=False, WB=None, **kwargs) -> MotionCommand:
        """
        计算运动指令
        Args:
            current_pose_px: 当前位姿 (x_px, y_px, yaw_rad)
            path_xy: 路径点列表 [(x, y), ...]
            target_speed: 目标速度 m/s
            current_speed: 当前速度 m/s
            return_steering: 是否返回转向角（阿克曼模型用）
            WB: 车辆轴距 (Wheelbase)，用于计算转向角
        """
        
        if self.mpc_controls_speed:
            # MPC同时控制速度和角速度
            result = self.angular_controller.compute_angular_control(
                current_pose_px, path_xy, target_speed, current_speed=current_speed, WB=WB, **kwargs
            )
            if len(result) == 3:  # (v, steering_angle, omega)
                controlled_speed, steering_angle, omega = result
            else:
                omega, steering_angle = result
                controlled_speed = target_speed
        else:
            # 分离控制：先计算速度，再计算角度
            if self.speed_controller is not None:
                controlled_speed = self.speed_controller.compute_speed(
                    target_speed, **kwargs
                )
            else:
                controlled_speed = target_speed
                
            omega, steering_angle = self.angular_controller.compute_angular_control(
                current_pose_px, path_xy, controlled_speed, 
                return_steering=return_steering, WB=WB,**kwargs
            )

        if return_steering and steering_angle is not None:
            return MotionCommand(
                v=controlled_speed,
                steering_angle=steering_angle,
                command_type="steering"
            )
        else:
            return MotionCommand(
                v=controlled_speed,
                omega=omega if omega is not None else 0.0,
                command_type="velocity"
            )
    
