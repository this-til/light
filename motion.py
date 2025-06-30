from __future__ import annotations

import asyncio
import math
import time

import rospy
from geometry_msgs.msg import Twist

from command import CommandEvent
from main import Component, ConfigField
import util


class PIDController:
    """PID控制器类"""
    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0, max_output: float = float('inf')):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = None
    
    def reset(self):
        """重置PID控制器状态"""
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = None
    
    def compute(self, error: float) -> float:
        """计算PID输出"""
        current_time = time.time()
        
        if self.last_time is None:
            self.last_time = current_time
            self.previous_error = error
            return 0.0
        
        dt = current_time - self.last_time
        if dt <= 0:
            return 0.0
        
        # 比例项
        proportional = self.kp * error
        
        # 积分项
        self.integral += error * dt
        integral_term = self.ki * self.integral
        
        # 微分项
        derivative = (error - self.previous_error) / dt
        derivative_term = self.kd * derivative
        
        # 计算总输出
        output = proportional + integral_term + derivative_term
        
        # 限制输出范围
        if self.max_output != float('inf'):
            output = max(-self.max_output, min(self.max_output, output))
        
        # 更新历史值
        self.previous_error = error
        self.last_time = current_time
        
        return output


class MotionComponent(Component):
    speed: ConfigField[float] = ConfigField()
    turnSpeed: ConfigField[float] = ConfigField()
    # 旋转控制参数
    rotationTolerance: ConfigField[float] = ConfigField()  # 旋转容忍度（度）
    rotationKp: ConfigField[float] = ConfigField()  # 旋转比例控制系数
    maxRotationSpeed: ConfigField[float] = ConfigField()  # 最大旋转速度
    
    # PID控制参数
    pidKp: ConfigField[float] = ConfigField(default=0.02)  # PID比例系数
    pidKi: ConfigField[float] = ConfigField(default=0.001)  # PID积分系数
    pidKd: ConfigField[float] = ConfigField(default=0.005)  # PID微分系数

    velPub = None  # 添加发布器变量
    msg = None
    velocity: util.Velocity = None  # 使用封装的速度类型
    speedAttenuationEnabled = True  # 速度衰减开关，默认启用
    
    # PID控制器实例
    rotation_pid: PIDController = None

    async def awakeInit(self):
        await super().awakeInit()
        self.velPub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.msg = Twist()
        self.velocity = util.Velocity()  # 初始化速度对象
        
        # 初始化PID控制器
        self.rotation_pid = PIDController(
            kp=self.pidKp,
            ki=self.pidKi,
            kd=self.pidKd,
            max_output=self.maxRotationSpeed
        )

        asyncio.create_task(self.motionLoop())
        asyncio.create_task(self.speedAttenuationLoop())
        asyncio.create_task(self.instructionLoop())

    def setVelocity(self, velocity: util.Velocity = None, linear_x: float = None, linear_y: float = None, angular_z: float = None):
        """设置速度值的公共函数 - 支持 Velocity 对象和单独参数"""
        if velocity is not None:
            # 使用 Velocity 对象
            self.velocity = velocity
            self.msg.linear.x = velocity.linear.x
            self.msg.linear.y = velocity.linear.y
            self.msg.linear.z = velocity.linear.z
            self.msg.angular.x = velocity.angular.x
            self.msg.angular.y = velocity.angular.y
            self.msg.angular.z = velocity.angular.z
        else:
            # 使用单独参数（向后兼容）
            if linear_x is not None:
                self.velocity.linear.x = linear_x
                self.msg.linear.x = linear_x
            if linear_y is not None:
                self.velocity.linear.y = linear_y
                self.msg.linear.y = linear_y
            if angular_z is not None:
                self.velocity.angular.z = angular_z
                self.msg.angular.z = angular_z

        self.publishVelocity()
    
    def setVelocityFromComponents(self, linear: util.V3 = None, angular: util.V3 = None):
        """从向量组件设置速度"""
        velocity = util.Velocity(linear, angular)
        self.setVelocity(velocity)

    def publishVelocity(self):
        """发布速度消息的公共函数"""
        if self.velPub:
            self.velPub.publish(self.msg)

    def stopMotion(self):
        """停止所有运动的公共函数"""
        self.velocity.stop()
        self.setVelocity(self.velocity)

    def enableSpeedAttenuation(self):
        """启用速度衰减"""
        self.speedAttenuationEnabled = True
        self.logger.info("速度衰减已启用")

    def disableSpeedAttenuation(self):
        """禁用速度衰减"""
        self.speedAttenuationEnabled = False
        self.logger.info("速度衰减已禁用")

    def isSpeedAttenuationEnabled(self) -> bool:
        """检查速度衰减是否启用"""
        return self.speedAttenuationEnabled

    def setSpeedAttenuation(self, enabled: bool):
        """设置速度衰减状态"""
        if enabled:
            self.enableSpeedAttenuation()
        else:
            self.disableSpeedAttenuation()

    def getCurrentYaw(self) -> float:
        """获取当前偏航角（度）"""
        if hasattr(self.main, 'imuComponent'):
            return self.main.imuComponent.euler.z
        else:
            self.logger.warning("IMU组件不可用，无法获取当前角度")
            return 0.0

    def normalizeAngle(self, angle: float) -> float:
        """将角度规范化到[-180, 180]范围"""
        return util.normalizeAngleDegrees(angle)

    def calculateAngleDifference(self, target_angle: float, current_angle: float) -> float:
        """计算两个角度之间的最短角度差"""
        diff = target_angle - current_angle
        return self.normalizeAngle(diff)

    async def rotateToAngle(self, target_angle: float, timeout: float = 30.0, speed: float = None, use_pid: bool = False) -> bool:
        """
        旋转到指定的绝对角度
        
        参数:
            target_angle: 目标角度（度，-180到180）
            timeout: 超时时间（秒）
            speed: 旋转速度（rad/s），为None时使用默认turnSpeed
            use_pid: 是否启用PID控制，False时使用固定速度控制
        
        返回:
            bool: 是否成功到达目标角度
        """
        # 使用指定的速度或默认速度
        rotation_speed = speed if speed is not None else self.turnSpeed
        
        control_mode = "PID控制" if use_pid else f"固定速度控制 ({rotation_speed:.3f} rad/s)"
        self.logger.info(f"开始旋转到目标角度: {target_angle}°，控制模式: {control_mode}")
        
        # 规范化目标角度
        target_angle = self.normalizeAngle(target_angle)
        
        start_time = asyncio.get_event_loop().time()
        was_attenuation_enabled = self.speedAttenuationEnabled
        
        try:
            # 暂时禁用速度衰减
            self.disableSpeedAttenuation()
            
            # 如果使用PID控制，重置PID控制器
            if use_pid:
                self.rotation_pid.reset()
            
            loop_count = 0  # 循环计数器，用于控制日志输出频率
            
            while True:
                current_time = asyncio.get_event_loop().time()
                loop_count += 1
                
                # 检查超时
                if current_time - start_time > timeout:
                    self.logger.warning(f"旋转到角度 {target_angle}° 超时")
                    self.stopMotion()
                    return False
                
                # 获取当前角度
                current_angle = self.getCurrentYaw()
                
                # 计算角度误差
                angle_error = self.calculateAngleDifference(target_angle, current_angle)
                
                # 检查是否到达目标
                if abs(angle_error) <= self.rotationTolerance:
                    self.logger.info(f"成功旋转到目标角度: {target_angle}°，当前角度: {current_angle:.2f}°")
                    self.stopMotion()
                    return True
                
                # 计算控制输出
                if use_pid:
                    # 使用PID控制
                    # 将角度误差转换为弧度进行PID计算
                    angle_error_rad = math.radians(angle_error)
                    angular_velocity = self.rotation_pid.compute(angle_error_rad)
                    
                    # 确保最小旋转速度，避免在目标附近停滞
                    min_velocity = 0.05  # 最小角速度 rad/s
                    if abs(angular_velocity) < min_velocity and abs(angle_error) > self.rotationTolerance / 2:
                        angular_velocity = min_velocity if angle_error > 0 else -min_velocity
                else:
                    # 使用固定速度控制
                    if angle_error > 0:
                        angular_velocity = rotation_speed
                    else:
                        angular_velocity = -rotation_speed

                # 每5个循环输出一次详细日志（约0.25秒一次）
                if loop_count % 5 == 1:
                    elapsed_time = current_time - start_time
                    control_info = "PID" if use_pid else "固定速度"
                    self.logger.info(
                        f"旋转中({control_info}) - 当前: {current_angle:.2f}°, 目标: {target_angle}°, "
                        f"差距: {angle_error:.2f}°, 速度: {angular_velocity:.3f} rad/s, "
                        f"用时: {elapsed_time:.1f}s"
                    )
                
                # 设置旋转速度
                self.setVelocity(angular_z=angular_velocity)
                
                # 控制循环频率
                await asyncio.sleep(0.05)
                
        except asyncio.CancelledError:
            self.logger.info("旋转任务被取消")
            self.stopMotion()
            raise
        except Exception as e:
            self.logger.error(f"旋转到角度过程中发生错误: {e}")
            self.stopMotion()
            return False
        finally:
            # 恢复速度衰减设置
            self.setSpeedAttenuation(was_attenuation_enabled)
            self.stopMotion()

    async def rotateByAngle(self, delta_angle: float, timeout: float = 30.0, speed: float = None, use_pid: bool = False) -> bool:
        """
        相对当前位置旋转指定角度
        
        参数:
            delta_angle: 要旋转的角度（度，正值为逆时针，负值为顺时针）
            timeout: 超时时间（秒）
            speed: 旋转速度（rad/s），为None时使用默认turnSpeed
            use_pid: 是否启用PID控制，False时使用固定速度控制
        
        返回:
            bool: 是否成功完成旋转
        """
        rotation_speed = speed if speed is not None else self.turnSpeed
        control_mode = "PID控制" if use_pid else f"固定速度控制 ({rotation_speed:.3f} rad/s)"
        self.logger.info(f"开始相对旋转 {delta_angle}°，控制模式: {control_mode}")
        
        # 获取当前角度
        current_angle = self.getCurrentYaw()
        
        # 计算目标角度
        target_angle = current_angle + delta_angle
        target_angle_normalized = self.normalizeAngle(target_angle)
        
        self.logger.info(
            f"相对旋转计算 - 当前: {current_angle:.2f}°, 增量: {delta_angle:.2f}°, "
            f"目标: {target_angle:.2f}° → 规范化: {target_angle_normalized:.2f}°"
        )
        
        # 调用绝对角度旋转方法
        return await self.rotateToAngle(target_angle_normalized, timeout, speed, use_pid)

    async def rotateLeft(self, angle: float = 90.0, timeout: float = 30.0, speed: float = None, use_pid: bool = False) -> bool:
        """向左（逆时针）旋转指定角度"""
        rotation_speed = speed if speed is not None else self.turnSpeed
        control_mode = "PID控制" if use_pid else f"固定速度控制 ({rotation_speed:.3f} rad/s)"
        self.logger.info(f"向左旋转 {angle}° (逆时针)，控制模式: {control_mode}")
        return await self.rotateByAngle(angle, timeout, speed, use_pid)

    async def rotateRight(self, angle: float = 90.0, timeout: float = 30.0, speed: float = None, use_pid: bool = False) -> bool:
        """向右（顺时针）旋转指定角度"""
        rotation_speed = speed if speed is not None else self.turnSpeed
        control_mode = "PID控制" if use_pid else f"固定速度控制 ({rotation_speed:.3f} rad/s)"
        self.logger.info(f"向右旋转 {angle}° (顺时针)，控制模式: {control_mode}")
        return await self.rotateByAngle(-angle, timeout, speed, use_pid)

    def handleOperation(self, operation: str):
        """处理不同操作类型的公共函数"""
        if operation == "translationAdvance":
            # 向前平移
            velocity = util.Velocity.create(linear_x=self.speed)
            self.setVelocity(velocity)
        elif operation == "translationRetreat":
            # 向后平移
            velocity = util.Velocity.create(linear_x=-self.speed)
            self.setVelocity(velocity)
        elif operation == "translationLeft":
            # 左平移
            velocity = util.Velocity.create(linear_y=self.speed)
            self.setVelocity(velocity)
        elif operation == "translationRight":
            # 右平移
            velocity = util.Velocity.create(linear_y=-self.speed)
            self.setVelocity(velocity)
        elif operation == "angularLeft":
            # 左角速度+
            velocity = util.Velocity.create(angular_z=self.turnSpeed)
            self.setVelocity(velocity)
        elif operation == "angularRight":
            # 右角速度+
            velocity = util.Velocity.create(angular_z=-self.turnSpeed)
            self.setVelocity(velocity)
        elif operation == "stop":
            # 停止
            self.stopMotion()

    async def motionLoop(self):
        queue: asyncio.Queue[CommandEvent] = await self.main.commandComponent.commandEvent.subscribe(
            asyncio.Queue(maxsize=8))

        while True:
            try:
                command: CommandEvent = await queue.get()

                if command.key == "Operation":
                    self.handleOperation(command.value)
                    self.publishVelocity()
                    await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(f"motionLoop 引发异常: {e}")

    def hasVelocity(self) -> bool:
        """检查是否有速度的公共函数"""
        return self.velocity.hasVelocity()

    def attenuateVelocity(self, attenuation_factor: float = 0.85):
        """速度衰减的公共函数"""
        self.velocity.attenuate(attenuation_factor)
        # 同步到ROS消息
        self.msg.linear.x = self.velocity.linear.x
        self.msg.linear.y = self.velocity.linear.y
        self.msg.linear.z = self.velocity.linear.z
        self.msg.angular.x = self.velocity.angular.x
        self.msg.angular.y = self.velocity.angular.y
        self.msg.angular.z = self.velocity.angular.z

    async def speedAttenuationLoop(self):
        while True:
            try:
                # 每0.1秒检查一次速度衰减
                await asyncio.sleep(0.05)

                # 只有启用速度衰减时才执行衰减逻辑
                if self.speedAttenuationEnabled:
                    # 衰减因子，1秒内衰减到0，每0.1秒衰减10%
                    attenuation_factor = 0.85

                    # 检查是否有速度需要衰减
                    if self.hasVelocity():
                        self.attenuateVelocity(attenuation_factor)
                        self.publishVelocity()

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(f"speedAttenuationLoop 引发异常: {e}")

    async def motionTime(
            self,
            velocity: util.Velocity = None,
            linear_x: float = None,
            linear_y: float = None,
            angular_z: float = None,
            time: float = 1
    ):
        """运动指定时间 - 支持 Velocity 对象和单独参数"""
        try:
            self.disableSpeedAttenuation()
            if velocity is not None:
                self.setVelocity(velocity)
            else:
                self.setVelocity(linear_x=linear_x, linear_y=linear_y, angular_z=angular_z)
            await asyncio.sleep(time)
        finally:
            self.enableSpeedAttenuation()
            self.stopMotion()
    
    async def motionTimeWithComponents(self, linear: util.V3 = None, angular: util.V3 = None, time: float = 1):
        """使用向量组件运动指定时间"""
        velocity = util.Velocity(linear, angular)
        await self.motionTime(velocity=velocity, time=time)

    async def instructionLoop(self):
        """指令控制循环，处理运动相关的键盘指令"""
        queue = await self.main.keyComponent.keyEvent.subscribe(asyncio.Queue(maxsize=1))

        while True:
            try:
                key = await queue.get()
                
                self.logger.info(f"收到运动指令: {key}")

                # 旋转到指定角度指令：rotateToAngle:90 或 rotateToAngle:90:0.5 或 rotateToAngle:90:0.5:pid
                if key.startswith("rotateToAngle:"):
                    try:
                        parts = key.split(":")
                        angle_str = parts[1]
                        target_angle = float(angle_str)
                        speed = None
                        use_pid = False
                        
                        # 检查是否有指定速度参数
                        if len(parts) >= 3:
                            speed_str = parts[2]
                            speed = float(speed_str)
                            if speed <= 0 or speed > 5.0:
                                self.logger.warning(f"旋转速度超出范围: {speed} rad/s (有效范围: 0 ~ 5.0)")
                                continue
                        
                        # 检查是否启用PID控制
                        if len(parts) >= 4 and parts[3].lower() == "pid":
                            use_pid = True
                        
                        if -180 <= target_angle <= 180:
                            success = await self.rotateToAngle(target_angle, speed=speed, use_pid=use_pid)
                            speed_info = f"，速度: {speed:.3f} rad/s" if speed is not None else ""
                            control_info = f"，PID控制: {'启用' if use_pid else '禁用'}"
                            self.logger.info(f"旋转到角度 {target_angle}° {'成功' if success else '失败'}{speed_info}{control_info}")
                        else:
                            self.logger.warning(f"目标角度超出范围: {target_angle}° (有效范围: -180° ~ 180°)")
                    except (ValueError, IndexError) as e:
                        self.logger.error(f"无效的角度格式: {key}, 错误: {str(e)}")

                # 相对旋转指令：rotateBy:45 或 rotateBy:45:0.8 或 rotateBy:45:0.8:pid
                elif key.startswith("rotateBy:"):
                    try:
                        parts = key.split(":")
                        angle_str = parts[1]
                        delta_angle = float(angle_str)
                        speed = None
                        use_pid = False
                        
                        # 检查是否有指定速度参数
                        if len(parts) >= 3:
                            speed_str = parts[2]
                            speed = float(speed_str)
                            if speed <= 0 or speed > 5.0:
                                self.logger.warning(f"旋转速度超出范围: {speed} rad/s (有效范围: 0 ~ 5.0)")
                                continue
                        
                        # 检查是否启用PID控制
                        if len(parts) >= 4 and parts[3].lower() == "pid":
                            use_pid = True
                        
                        if -360 <= delta_angle <= 360:
                            success = await self.rotateByAngle(delta_angle, speed=speed, use_pid=use_pid)
                            speed_info = f"，速度: {speed:.3f} rad/s" if speed is not None else ""
                            control_info = f"，PID控制: {'启用' if use_pid else '禁用'}"
                            self.logger.info(f"相对旋转 {delta_angle}° {'成功' if success else '失败'}{speed_info}{control_info}")
                        else:
                            self.logger.warning(f"旋转角度超出范围: {delta_angle}° (有效范围: -360° ~ 360°)")
                    except (ValueError, IndexError) as e:
                        self.logger.error(f"无效的角度格式: {key}, 错误: {str(e)}")

                # 向左旋转指令：rotateLeft 或 rotateLeft:90 或 rotateLeft:90:0.5 或 rotateLeft:90:0.5:pid
                elif key == "rotateLeft":
                    success = await self.rotateLeft()  # 默认90度
                    self.logger.info(f"向左旋转90° {'成功' if success else '失败'}")
                elif key.startswith("rotateLeft:"):
                    try:
                        parts = key.split(":")
                        angle_str = parts[1]
                        angle = float(angle_str)
                        speed = None
                        use_pid = False
                        
                        # 检查是否有指定速度参数
                        if len(parts) >= 3:
                            speed_str = parts[2]
                            speed = float(speed_str)
                            if speed <= 0 or speed > 5.0:
                                self.logger.warning(f"旋转速度超出范围: {speed} rad/s (有效范围: 0 ~ 5.0)")
                                continue
                        
                        # 检查是否启用PID控制
                        if len(parts) >= 4 and parts[3].lower() == "pid":
                            use_pid = True
                        
                        if 0 < angle <= 360:
                            success = await self.rotateLeft(angle, speed=speed, use_pid=use_pid)
                            speed_info = f"，速度: {speed:.3f} rad/s" if speed is not None else ""
                            control_info = f"，PID控制: {'启用' if use_pid else '禁用'}"
                            self.logger.info(f"向左旋转 {angle}° {'成功' if success else '失败'}{speed_info}{control_info}")
                        else:
                            self.logger.warning(f"旋转角度超出范围: {angle}° (有效范围: 0° ~ 360°)")
                    except (ValueError, IndexError) as e:
                        self.logger.error(f"无效的角度格式: {key}, 错误: {str(e)}")

                # 向右旋转指令：rotateRight 或 rotateRight:45 或 rotateRight:45:0.3 或 rotateRight:45:0.3:pid
                elif key == "rotateRight":
                    success = await self.rotateRight()  # 默认90度
                    self.logger.info(f"向右旋转90° {'成功' if success else '失败'}")
                elif key.startswith("rotateRight:"):
                    try:
                        parts = key.split(":")
                        angle_str = parts[1]
                        angle = float(angle_str)
                        speed = None
                        use_pid = False
                        
                        # 检查是否有指定速度参数
                        if len(parts) >= 3:
                            speed_str = parts[2]
                            speed = float(speed_str)
                            if speed <= 0 or speed > 5.0:
                                self.logger.warning(f"旋转速度超出范围: {speed} rad/s (有效范围: 0 ~ 5.0)")
                                continue
                        
                        # 检查是否启用PID控制
                        if len(parts) >= 4 and parts[3].lower() == "pid":
                            use_pid = True
                        
                        if 0 < angle <= 360:
                            success = await self.rotateRight(angle, speed=speed, use_pid=use_pid)
                            speed_info = f"，速度: {speed:.3f} rad/s" if speed is not None else ""
                            control_info = f"，PID控制: {'启用' if use_pid else '禁用'}"
                            self.logger.info(f"向右旋转 {angle}° {'成功' if success else '失败'}{speed_info}{control_info}")
                        else:
                            self.logger.warning(f"旋转角度超出范围: {angle}° (有效范围: 0° ~ 360°)")
                    except (ValueError, IndexError) as e:
                        self.logger.error(f"无效的角度格式: {key}, 错误: {str(e)}")

                # 运动控制指令：motionTime:forward:2 (前进2秒)
                elif key.startswith("motionTime:"):
                    try:
                        parts = key.split(":")
                        if len(parts) >= 3:
                            direction = parts[1]
                            time_str = parts[2]
                            motion_time = float(time_str)
                            
                            if 0.1 <= motion_time <= 10.0:  # 限制合理的时间范围
                                if direction == "forward":
                                    await self.motionTime(linear_x=self.speed, time=motion_time)
                                    self.logger.info(f"前进 {motion_time}秒 完成")
                                elif direction == "backward":
                                    await self.motionTime(linear_x=-self.speed, time=motion_time)
                                    self.logger.info(f"后退 {motion_time}秒 完成")
                                elif direction == "left":
                                    await self.motionTime(linear_y=self.speed, time=motion_time)
                                    self.logger.info(f"左移 {motion_time}秒 完成")
                                elif direction == "right":
                                    await self.motionTime(linear_y=-self.speed, time=motion_time)
                                    self.logger.info(f"右移 {motion_time}秒 完成")
                                elif direction == "rotateLeft":
                                    await self.motionTime(angular_z=self.turnSpeed, time=motion_time)
                                    self.logger.info(f"左转 {motion_time}秒 完成")
                                elif direction == "rotateRight":
                                    await self.motionTime(angular_z=-self.turnSpeed, time=motion_time)
                                    self.logger.info(f"右转 {motion_time}秒 完成")
                                else:
                                    self.logger.warning(f"未知的运动方向: {direction}")
                            else:
                                self.logger.warning(f"运动时间超出范围: {motion_time}秒 (有效范围: 0.1s ~ 10.0s)")
                        else:
                            self.logger.error(f"无效的运动时间指令格式: {key}")
                    except (ValueError, IndexError) as e:
                        self.logger.error(f"无效的运动时间格式: {key}, 错误: {str(e)}")

                # 停止运动指令
                elif key == "stopMotion":
                    self.stopMotion()
                    self.logger.info("已停止所有运动")

                # 获取当前角度信息指令
                elif key == "getCurrentAngle":
                    if hasattr(self.main, 'imuComponent'):
                        current_yaw = self.main.imuComponent.getYaw()
                        current_euler = self.main.imuComponent.getEulerAngles()
                        self.logger.info(f"当前角度信息 - 偏航角: {current_yaw:.2f}°, 欧拉角: {current_euler}")
                    else:
                        self.logger.warning("IMU组件不可用，无法获取角度信息")

                # 速度衰减控制指令
                elif key == "enableSpeedAttenuation":
                    self.enableSpeedAttenuation()
                elif key == "disableSpeedAttenuation":
                    self.disableSpeedAttenuation()

                # 调整旋转控制参数指令
                elif key.startswith("setRotationTolerance:"):
                    try:
                        value_str = key.split(":")[1]
                        tolerance = float(value_str)
                        if 0.1 <= tolerance <= 10.0:
                            self.rotationTolerance = tolerance
                            self.logger.info(f"旋转容忍度已设置为: {tolerance}°")
                        else:
                            self.logger.warning(f"旋转容忍度超出范围: {tolerance}° (有效范围: 0.1° ~ 10.0°)")
                    except (ValueError, IndexError) as e:
                        self.logger.error(f"无效的容忍度格式: {key}, 错误: {str(e)}")

                elif key.startswith("setRotationKp:"):
                    try:
                        value_str = key.split(":")[1]
                        kp = float(value_str)
                        if 0.001 <= kp <= 0.1:
                            self.rotationKp = kp
                            self.logger.info(f"旋转比例系数已设置为: {kp}")
                        else:
                            self.logger.warning(f"比例系数超出范围: {kp} (有效范围: 0.001 ~ 0.1)")
                    except (ValueError, IndexError) as e:
                        self.logger.error(f"无效的比例系数格式: {key}, 错误: {str(e)}")

                elif key.startswith("setMaxRotationSpeed:"):
                    try:
                        value_str = key.split(":")[1]
                        max_speed = float(value_str)
                        if 0.1 <= max_speed <= 3.0:
                            self.maxRotationSpeed = max_speed
                            # 同时更新PID控制器的最大输出
                            self.rotation_pid.max_output = max_speed
                            self.logger.info(f"最大旋转速度已设置为: {max_speed} rad/s")
                        else:
                            self.logger.warning(f"最大旋转速度超出范围: {max_speed} (有效范围: 0.1 ~ 3.0)")
                    except (ValueError, IndexError) as e:
                        self.logger.error(f"无效的最大速度格式: {key}, 错误: {str(e)}")

                # PID参数调整指令
                elif key.startswith("setPidKp:"):
                    try:
                        value_str = key.split(":")[1]
                        kp = float(value_str)
                        if 0.001 <= kp <= 0.5:
                            self.pidKp = kp
                            self.rotation_pid.kp = kp
                            self.logger.info(f"PID比例系数已设置为: {kp}")
                        else:
                            self.logger.warning(f"PID Kp超出范围: {kp} (有效范围: 0.001 ~ 0.5)")
                    except (ValueError, IndexError) as e:
                        self.logger.error(f"无效的PID Kp格式: {key}, 错误: {str(e)}")

                elif key.startswith("setPidKi:"):
                    try:
                        value_str = key.split(":")[1]
                        ki = float(value_str)
                        if 0.0 <= ki <= 0.1:
                            self.pidKi = ki
                            self.rotation_pid.ki = ki
                            self.logger.info(f"PID积分系数已设置为: {ki}")
                        else:
                            self.logger.warning(f"PID Ki超出范围: {ki} (有效范围: 0.0 ~ 0.1)")
                    except (ValueError, IndexError) as e:
                        self.logger.error(f"无效的PID Ki格式: {key}, 错误: {str(e)}")

                elif key.startswith("setPidKd:"):
                    try:
                        value_str = key.split(":")[1]
                        kd = float(value_str)
                        if 0.0 <= kd <= 0.1:
                            self.pidKd = kd
                            self.rotation_pid.kd = kd
                            self.logger.info(f"PID微分系数已设置为: {kd}")
                        else:
                            self.logger.warning(f"PID Kd超出范围: {kd} (有效范围: 0.0 ~ 0.1)")
                    except (ValueError, IndexError) as e:
                        self.logger.error(f"无效的PID Kd格式: {key}, 错误: {str(e)}")
                
                # 重置PID控制器指令
                elif key == "resetPid":
                    self.rotation_pid.reset()
                    self.logger.info("PID控制器已重置")

                # 显示当前参数状态
                elif key == "showMotionParams":
                    self.logger.info("=== 当前运动参数 ===")
                    self.logger.info(f"基础速度: {self.speed} m/s")
                    self.logger.info(f"转弯速度: {self.turnSpeed} rad/s")
                    self.logger.info(f"旋转容忍度: {self.rotationTolerance}°")
                    self.logger.info(f"旋转比例系数: {self.rotationKp}")
                    self.logger.info(f"最大旋转速度: {self.maxRotationSpeed} rad/s")
                    self.logger.info(f"PID参数 - Kp: {self.pidKp}, Ki: {self.pidKi}, Kd: {self.pidKd}")
                    self.logger.info(f"速度衰减: {'启用' if self.speedAttenuationEnabled else '禁用'}")
                    if hasattr(self.main, 'imuComponent'):
                        current_yaw = self.main.imuComponent.getYaw()
                        self.logger.info(f"当前偏航角: {current_yaw:.2f}°")
                    self.logger.info("==================")

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(f"运动指令处理异常: {str(e)}")
