from abc import ABC, abstractmethod
from utils import PIControl


class SpeedController(ABC):
    """速度控制器抽象基类"""
    @abstractmethod
    def compute_speed(self, target_speed, current_speed, **kwargs) -> float:
        """计算控制速度"""
        pass

class PISpeedController(SpeedController):
    """PI速度控制器"""
    def __init__(self, kp=5.0, ki=0.5, n=5, max_acc=10, min_acc=-10):
        self.controller = PIControl(kp, ki, n, max_acc, min_acc)
        self.current_speed = 0.0
        
    def compute_speed(self, target_speed, current_speed=None, dt=0.05, **kwargs) -> float:
        if current_speed is not None:
            self.current_speed = current_speed
            
        speed_error = target_speed - self.current_speed
        acceleration = self.controller.control(speed_error)
        self.current_speed += acceleration * dt
        
        return self.current_speed

class ConstantSpeedController(SpeedController):
    """恒定速度控制器"""
    def __init__(self, constant_speed=1.0):
        self.constant_speed = constant_speed
        
    def compute_speed(self, **kwargs) -> float:
        return self.constant_speed