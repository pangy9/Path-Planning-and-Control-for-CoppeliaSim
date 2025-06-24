import collections
import math
import numpy as np

class PIControl:
    def __init__(self, kp, ki, n, max_signal, min_signal):
        self.kp = kp
        self.ki = ki
        self.n = n # 积分窗口大小
        self.max_signal = max_signal
        self.min_signal = min_signal
        self.integral_window = collections.deque(maxlen=n)

    def control(self, error):
        self.integral_window.append(error)
        
        integral_sum = sum(self.integral_window)

        # P项
        p_term = self.kp * error
        # I项 (ki * integral)
        i_term = self.ki * integral_sum
        
        unbounded_signal = p_term + i_term
        signal = max(min(unbounded_signal, self.max_signal), self.min_signal)

        return signal


def smooth(path, weight_data=0.5, weight_smooth=0.1, tolerance=0.000001):
    """
    创建一个平滑的n维坐标路径。
    参数:
        path: 包含 (x,y) 坐标的路径列表
        weight_data: 浮点数，数据点更新的权重 (alpha)
        weight_smooth: 浮点数，平滑坐标的权重 (beta)。
        tolerance: 浮点数，每次迭代所需的最小变化量，小于此值则停止迭代。
    输出:
        new: 包含平滑坐标的列表。 (x,y) 格式
    """

    new = [list(p) for p in path] # 创建可变副本
    dims = len(path[0]) # 维度数量 (例如，x,y为2)
    change = tolerance # 初始化变化量，确保第一次迭代运行

    while change >= tolerance:
        change = 0.0
        for i in range(1, len(new) - 1): # 排除起点和终点
            for j in range(dims):
                x_i = path[i][j] # 原始数据点
                y_i, y_prev, y_next = new[i][j], new[i - 1][j], new[i + 1][j] # 当前、前一个、后一个平滑点

                y_i_saved = y_i # 保存当前平滑值
                # 更新规则进行平滑
                y_i += weight_data * (x_i - y_i) + weight_smooth * (y_next + y_prev - (2 * y_i))
                new[i][j] = y_i

                change += abs(y_i - y_i_saved) # 累积变化量

    return new

def smooth_path_yaw(yaw_angles):
    """
    平滑路径航向角，避免角度跳跃
    
    航向角可能从359°跳到1°，数值上差很大但实际相近
    """
    for i in range(len(yaw_angles) - 1):
        yaw_difference = yaw_angles[i + 1] - yaw_angles[i]
        
        # 处理角度跳跃
        while yaw_difference >= math.pi / 2.0:
            yaw_angles[i + 1] -= math.pi * 2.0
            yaw_difference = yaw_angles[i + 1] - yaw_angles[i]
        
        while yaw_difference <= -math.pi / 2.0:
            yaw_angles[i + 1] += math.pi * 2.0
            yaw_difference = yaw_angles[i + 1] - yaw_angles[i]
    
    return yaw_angles

def angle_mod(x, zero_2_2pi=False, degree=False):
    """
    Angle modulo operation
    Default angle modulo range is [-pi, pi)

    Parameters
    ----------
    x : float or array_like
        A angle or an array of angles. This array is flattened for
        the calculation. When an angle is provided, a float angle is returned.
    zero_2_2pi : bool, optional
        Change angle modulo range to [0, 2pi)
        Default is False.
    degree : bool, optional
        If True, then the given angles are assumed to be in degrees.
        Default is False.

    Returns
    -------
    ret : float or ndarray
        an angle or an array of modulated angle.

    Examples
    --------
    >>> angle_mod(-4.0)
    2.28318531

    >>> angle_mod([-4.0])
    np.array(2.28318531)

    >>> angle_mod([-150.0, 190.0, 350], degree=True)
    array([-150., -170.,  -10.])

    >>> angle_mod(-60.0, zero_2_2pi=True, degree=True)
    array([300.])

    """
    if isinstance(x, float):
        is_float = True
    else:
        is_float = False

    x = np.asarray(x).flatten()
    if degree:
        x = np.deg2rad(x)

    if zero_2_2pi:
        mod_angle = x % (2 * np.pi)
    else:
        mod_angle = (x + np.pi) % (2 * np.pi) - np.pi

    if degree:
        mod_angle = np.rad2deg(mod_angle)

    if is_float:
        return mod_angle.item()
    else:
        return mod_angle