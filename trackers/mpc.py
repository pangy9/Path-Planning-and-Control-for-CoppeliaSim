import numpy as np
import math
import matplotlib.pyplot as plt
import cvxpy
# 设置matplotlib的字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from utils import smooth_path_yaw, angle_mod
from .angular import AngularController

class DifferentialDriveMPCAngularController(AngularController):
    """两轮差分MPC跟踪器（带可视化）"""
    def __init__(
        self,
        pixels_to_meters_scale,
        state_weight,
        control_weight,
        terminal_weight,
        control_change_weight,
        control_speed=False,
        ref_speed=None,
        horizon_length=5,
        dt=0.05,
        MAX_V=1.5,
        MAX_ACCELERATION=2.0,
        MAX_OMEGA=3.0,
        MAX_OMEGA_CHANGE=0.5,
    ):
        # MPC参数
        self.state_weight = state_weight
        self.control_weight = control_weight
        self.terminal_weight = terminal_weight
        self.control_change_weight = control_change_weight
        self.control_speed = control_speed # 是否控制速度
        self.ref_speed = ref_speed
        self.horizon_length = horizon_length
        
        self.pixels_to_meters_scale = pixels_to_meters_scale
        self.dt = dt
        
        self._target_yaws = None  # 缓存路径点的目标航向角
        self.prev_acc_opts = [0.0] * self.horizon_length
        self.prev_omega_opts = [0.0] * self.horizon_length
        
        # 约束
        self.MAX_V = MAX_V
        self.MAX_ACCELERATION = MAX_ACCELERATION
        self.MAX_OMEGA = MAX_OMEGA
        self.MAX_OMEGA_CHANGE = MAX_OMEGA_CHANGE

        self.current_speed = None  # 当前实际速度
                
        # 可视化相关
        self.debug_data = {
            'predicted_trajectory': None,
            'reference_trajectory': None,
            'current_state': None,
            'control_output': None,
            'solver_status': None
        }
        
    def compute_angular_control(self, current_pose_px, path_xy, target_speed, current_speed=None, return_steering=False, WB=None, display_image=None, **kwargs):

        if current_speed is not None:
            self.current_speed = current_speed
        else:
            self.current_speed = target_speed
            
        if self._target_yaws is None or len(self._target_yaws) != len(path_xy):
            self._target_yaws = smooth_path_yaw(self._calculate_target_yaws(path_xy))

        # 转换坐标系
        current_pos_m = np.array(current_pose_px[:2]) * self.pixels_to_meters_scale
        current_yaw = current_pose_px[2]
        self.prev_yaw = current_yaw
        
        # 构建参考轨迹
        xref = self._build_reference_trajectory(target_speed, current_pos_m, path_xy)
        
        # 当前状态 [x, y, yaw, v]
        x0 = np.array([current_pos_m[0], current_pos_m[1], current_yaw, self.current_speed])
        
        # 保存调试数据
        self.debug_data['current_state'] = x0
        self.debug_data['reference_trajectory'] = xref
        
        # 求解MPC优化问题
        self.prev_acc_opts, self.prev_omega_opts, x_opt = self._iterative_mpc_control(x0, xref)
        acc_opt = self.prev_acc_opts[0] if self.prev_acc_opts is not None else 0.0
        omega_opt = self.prev_omega_opts[0] if self.prev_omega_opts is not None else 0.0
        
        # 保存预测轨迹和控制输出
        self.debug_data['predicted_trajectory'] = x_opt
        self.debug_data['control_output'] = acc_opt, omega_opt
        
        next_speed = self.current_speed + acc_opt * self.dt
        next_speed = np.clip(next_speed, -self.MAX_V, self.MAX_V)  # 速度限制
        self.current_speed = next_speed
          
        steering = None
        if return_steering:
            # 如果需要返回转向角（阿克曼模型）
            steering = math.atan2(WB * omega_opt, next_speed) if next_speed != 0 else 0.0
            
        # 可视化
        if display_image is not None:
            self._visualize_mpc(current_pose_px, path_xy, display_image)
        
        if self.control_speed:
            return next_speed, steering, omega_opt
        else:
            return omega_opt, steering
        
    def _get_linearized_model_matrices(self, yaw, speed):
        """线性化运动学模型"""
        A = np.eye(4)
        A[0, 2] = - self.dt * speed * math.sin(yaw)
        A[1, 2] = self.dt * speed * math.cos(yaw)
        A[0, 3] = self.dt * math.cos(yaw)
        A[1, 3] = self.dt * math.sin(yaw)
        
        B = np.zeros((4, 2))
        B[2, 1] = self.dt
        B[3, 0] = self.dt
        
        C = np.zeros(4)
        C[0] = self.dt * speed * math.sin(yaw) * yaw
        C[1] = - self.dt * speed * math.cos(yaw) * yaw
        
        return A, B, C
    
    def _predict_vehicle_trajectory(self, x0, omega_opts, acc_opts):
        """
        预测车辆轨迹 - 根据当前状态和控制序列预测未来轨迹
        
        这里使用非线性模型进行预测，得到的轨迹更接近真实情况
        Args:
            x0 (np.ndarray): 初始状态 [x, y, v, yaw]
            omega_opts (list): 优化得到的角速度序列
            v_opts (list): 优化得到的线速度序列
        """
        x_trajectory = np.zeros((4, self.horizon_length + 1))
        x_trajectory[:, 0] = x0
        
        for i in range(self.horizon_length):
            x, y, yaw, v = x_trajectory[:, i]
            acceleration = acc_opts[i]
            omega = omega_opts[i]
            
            x_trajectory[0, i + 1] = x + self.dt * v * math.cos(yaw)
            x_trajectory[1, i + 1] = y + self.dt * v * math.sin(yaw)
            x_trajectory[2, i + 1] = yaw + self.dt * omega
            x_trajectory[3, i + 1] = v + self.dt * acceleration
            
            # 速度限制
            x_trajectory[3, i + 1] = np.clip(x_trajectory[3, i + 1], -self.MAX_V, self.MAX_V)
        
        return x_trajectory

    def _normalize_angle_continuous(self, angle, prev_angle=None):
        """
        连续的航向角归一化 - 避免角度跳跃
        """
        if prev_angle is None:
            return angle_mod(angle)
        
        # 计算与前一个角度的差值
        diff = angle - prev_angle
        
        # 如果差值超过π，说明发生了跳跃，需要调整
        if diff > math.pi:
            angle -= 2 * math.pi
        elif diff < -math.pi:
            angle += 2 * math.pi
            
        return angle
       
    def _build_reference_trajectory(self, target_speed, current_pos_m, path_xy):
        path_m = [(p[0] * self.pixels_to_meters_scale, p[1] * self.pixels_to_meters_scale) for p in path_xy]
        distances = [np.linalg.norm(current_pos_m - np.array(p)) for p in path_m]
        closest_idx = np.argmin(distances)
        
        xref = np.zeros((4, self.horizon_length + 1))
        
        # 为预测时域生成参考轨迹
        travel = 0.0
        # 每个路径点之间的距离
        path_resolution = self.pixels_to_meters_scale
        for i in range(self.horizon_length + 1):
            travel += self.dt * abs(target_speed)
            ref_idx = min(int(round(closest_idx + travel / path_resolution)), len(path_m) - 1)
            xref[0, i] = path_m[ref_idx][0]  # x
            xref[1, i] = path_m[ref_idx][1]  # y
            
            # 连续的航向角处理
            target_yaw = self._target_yaws[ref_idx]
            if i == 0:
                xref[2, i] = self._normalize_angle_continuous(target_yaw, self.prev_yaw)
                self.prev_yaw = xref[2, i]
            else:
                xref[2, i] = self._normalize_angle_continuous(target_yaw, xref[2, i-1])
            
            xref[3, i] = target_speed

        return xref
    
    def _calculate_target_yaws(self, path_xy):
        """
        计算并缓存路径上每个点的目标航向角。
        """
        target_yaws = []
        n_points = len(path_xy)
        for i in range(n_points - 1):
            # 寻找当前点前方的一个点用于计算方向
            # 如果是路径末尾，则使用最后一个路径段的方向
            p_ahead_idx = min(i + 1, n_points - 1)
            
            dx = path_xy[p_ahead_idx][0] - path_xy[i][0]
            dy = path_xy[p_ahead_idx][1] - path_xy[i][1]
            
            target_yaw = math.atan2(dy, dx)
            target_yaws.append(target_yaw)
        target_yaws.append(target_yaws[-1])
        return target_yaws
    
    def _linear_mpc_optim(self, xref, x_predicted, x0):
        # 决策变量
        x = cvxpy.Variable((4, self.horizon_length + 1)) # 状态变量 [x, y, yaw, v]
        u = cvxpy.Variable((2, self.horizon_length)) # 控制变量 [acc, omega]
        cost = 0.0
        constraints = []
        
        goal_reached_index = None
        ref_diff = np.diff(xref, axis=1)
        for i in range(self.horizon_length):
            if all(ref_diff[:,i] == 0):
                goal_reached_index = i
                break
        
        for t in range(self.horizon_length):
            # 1. 控制输入代价 - 惩罚大的控制输入
            cost += cvxpy.quad_form(u[:, t], self.control_weight)

            # 2. 状态跟踪代价
            if goal_reached_index is not None and t >= goal_reached_index:
                cost += cvxpy.quad_form(xref[:, t] - x[:, t], self.state_weight)
            else:
                # 到达终点后：只约束位置
                position_error = xref[:2, t] - x[:2, t]
                
                # 位置误差权重可以增大，确保精确到达终点
                position_weight = self.state_weight[:2, :2] * 2.0
                
                cost += cvxpy.quad_form(position_error, position_weight)
            
            # 3. 动态约束 - 确保状态按照车辆模型演化
            A, B, C = self._get_linearized_model_matrices(x_predicted[2, t], x_predicted[3, t])
            constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]
                
            # 4. 控制平滑性代价 - 惩罚控制的剧烈变化
            if t < (self.horizon_length - 1):
                control_change = u[:, t + 1] - u[:, t]
                cost += cvxpy.quad_form(control_change, self.control_change_weight)
        
        # 5. 终端代价 - 最后一个状态的跟踪误差
        cost += cvxpy.quad_form(xref[:, self.horizon_length] - x[:, self.horizon_length], self.terminal_weight)
        
        # 6. 约束条件
        constraints += [x[:, 0] == x0]
        if self.control_speed:
            constraints += [cvxpy.abs(u[0, :]) <= self.MAX_ACCELERATION]
            constraints += [cvxpy.abs(x[3, :]) <= self.MAX_V]
        else:
            constraints += [u[0, :] == 0.0]
            
        constraints += [cvxpy.abs(u[1, :]) <= self.MAX_OMEGA]
        for t in range(self.horizon_length - 1):
            omega_change = u[1, t + 1] - u[1, t]
            constraints += [omega_change <= self.MAX_OMEGA_CHANGE]
            constraints += [omega_change >= -self.MAX_OMEGA_CHANGE]        
        
        # 求解
        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        prob.solve(solver=cvxpy.CLARABEL, verbose=False)
        
        self.debug_data['solver_status'] = prob.status
        
        if prob.status in [cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE]:
            # 返回角速度和预测轨迹
            return u.value[0], u.value[1], x[:, :self.horizon_length + 1].value
        else:
            print(f"MPC solver failed with status: {prob.status}")
            return None, None, None
                
            
    def _iterative_mpc_control(self, x0, xref):
        """
        迭代线性MPC控制器
        Args:
            x0 (np.ndarray): 初始状态 [x, y, yaw, v]
            xref (np.ndarray): 参考轨迹 [x, y, yaw, v]，形状为 (4, horizon_length + 1)
        """
        max_iterations = 3  # 最大迭代次数
        omega_opts = self.prev_omega_opts[:]
        acc_opts = self.prev_acc_opts[:]
        
        for i in range(max_iterations):
            # 1. 用非线性模型预测轨迹
            x_predicted = self._predict_vehicle_trajectory(x0, omega_opts, acc_opts)
            
            # 2. 保存上一次的控制序列
            _acc_opt = acc_opts[:]
            _omega_opt = omega_opts[:]
            
            # 3. 在预测轨迹附近线性化并求解优化问题
            acc_opts, omega_opts, x_opt = self._linear_mpc_optim(xref, x_predicted, x0)
            
            # 4. 检查收敛性
            du = np.sum(abs(_acc_opt - acc_opts)) + np.sum(abs(_omega_opt - omega_opts))
            if du <= 0.0005:  # 如果控制变化很小，认为收敛
                break
        else:
            print("Iterative is max iter")
            
        return acc_opts, omega_opts, x_opt

                
    def _visualize_mpc(self, current_pose_px, path_xy, display_image):
        """可视化MPC状态"""
        plt.cla()
        plt.imshow(display_image, origin='lower')
        
        # 绘制原始路径
        path_x = [p[0] for p in path_xy]
        path_y = [p[1] for p in path_xy]
        plt.plot(path_x, path_y, 'g-', linewidth=2, label='规划路径', alpha=0.7)
        
        # 绘制当前位置和朝向
        current_x, current_y, current_yaw = current_pose_px
        plt.scatter(current_x, current_y, c='blue', s=30, label='当前位置', zorder=5)
        arrow_length = 20
        arrow_dx = arrow_length * math.cos(current_yaw)
        arrow_dy = arrow_length * math.sin(current_yaw)
        plt.arrow(current_x, current_y, arrow_dx, arrow_dy, 
                 head_width=8, head_length=8, fc='blue', ec='blue', alpha=0.8)
        
        # 绘制参考轨迹（米坐标转像素坐标）
        if self.debug_data['reference_trajectory'] is not None:
            ref_traj = self.debug_data['reference_trajectory']
            ref_x_px = ref_traj[0, :] / self.pixels_to_meters_scale
            ref_y_px = ref_traj[1, :] / self.pixels_to_meters_scale
            
            plt.plot(ref_x_px, ref_y_px, 'r--', linewidth=2, label='MPC参考轨迹', alpha=0.8)
            plt.scatter(ref_x_px, ref_y_px, c='red', s=10, alpha=0.8, zorder=4)
        
        # 绘制预测轨迹
        if self.debug_data['predicted_trajectory'] is not None:
            pred_traj = self.debug_data['predicted_trajectory']
            pred_x_px = pred_traj[0, :] / self.pixels_to_meters_scale
            pred_y_px = pred_traj[1, :] / self.pixels_to_meters_scale
            
            plt.plot(pred_x_px, pred_y_px, 'y--', linewidth=2, label='MPC预测轨迹', alpha=0.9)
            plt.scatter(pred_x_px, pred_y_px, c='yellow', s=10, alpha=0.9, zorder=4)
        
        # 添加调试信息文本
        debug_text = self._generate_debug_text()
        plt.text(10, display_image.shape[0] - 10, debug_text, 
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.title('MPC路径跟踪调试', fontsize=12)
        plt.legend(loc='upper right')
        plt.pause(0.01)
    
    def _generate_debug_text(self):
        """生成调试信息文本"""
        debug_lines = []
        
        # 当前状态信息
        if self.debug_data['current_state'] is not None:
            x, y, yaw, v= self.debug_data['current_state']
            debug_lines.append(f"当前状态:")
            debug_lines.append(f"  位置: ({x:.2f}, {y:.2f})m")
            debug_lines.append(f"  航向: {yaw:.1f} rad")
            debug_lines.append(f"  速度: {v:.2f} m/s")
        
        # 控制输出
        if self.debug_data['control_output'] is not None:
            v, omega = self.debug_data['control_output']
            debug_lines.append(f"MPC输出:")
            debug_lines.append(f"  加速度: {v:.2f}m/s")
            debug_lines.append(f"  角速度: {omega:.3f} rad/s")
        
        # 预测时域信息
        debug_lines.append(f"预测时域: {self.horizon_length}步")
        debug_lines.append(f"时间步长: {self.dt}s")
        
        return '\n'.join(debug_lines)
