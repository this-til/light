from __future__ import annotations

import asyncio
import math

import rospy
from geometry_msgs.msg import Twist

from command import CommandEvent
from main import Component, ConfigField
import util


class MotionComponent(Component):
    speed: ConfigField[float] = ConfigField()
    turnSpeed: ConfigField[float] = ConfigField()
    # 旋转控制参数
    rotationTolerance: ConfigField[float] = ConfigField()  # 旋转容忍度（度）
    rotationKp: ConfigField[float] = ConfigField()  # 旋转比例控制系数
    maxRotationSpeed: ConfigField[float] = ConfigField()  # 最大旋转速度

    velPub = None  # 添加发布器变量
    msg = None
    velocity: util.Velocity = None  # 使用封装的速度类型
    speedAttenuationEnabled = True  # 速度衰减开关，默认启用

    async def awakeInit(self):
        await super().awakeInit()
        self.velPub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.msg = Twist()
        self.velocity = util.Velocity()  # 初始化速度对象

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

    async def rotateToAngle(self, target_angle: float, timeout: float = 30.0) -> bool:
        """
        旋转到指定的绝对角度
        
        参数:
            target_angle: 目标角度（度，-180到180）
            timeout: 超时时间（秒）
        
        返回:
            bool: 是否成功到达目标角度
        """
        self.logger.info(f"开始旋转到目标角度: {target_angle}°")
        
        # 规范化目标角度
        target_angle = self.normalizeAngle(target_angle)
        
        start_time = asyncio.get_event_loop().time()
        was_attenuation_enabled = self.speedAttenuationEnabled
        
        try:
            # 暂时禁用速度衰减
            self.disableSpeedAttenuation()
            
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
                
                # 计算控制输出（比例控制）
                #angular_velocity = angle_error * self.rotationKp

                # 限制最大旋转速度
                #angular_velocity = util.clamp(angular_velocity, -self.maxRotationSpeed, self.maxRotationSpeed)

                if angle_error > 0:
                    angular_velocity = self.turnSpeed
                else:
                    angular_velocity = -self.turnSpeed

                # 每5个循环输出一次详细日志（约0.25秒一次）
                if loop_count % 5 == 1:
                    elapsed_time = current_time - start_time
                    self.logger.info(
                        f"旋转中 - 当前: {current_angle:.2f}°, 目标: {target_angle}°, "
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

    async def rotateByAngle(self, delta_angle: float, timeout: float = 30.0) -> bool:
        """
        相对当前位置旋转指定角度
        
        参数:
            delta_angle: 要旋转的角度（度，正值为逆时针，负值为顺时针）
            timeout: 超时时间（秒）
        
        返回:
            bool: 是否成功完成旋转
        """
        self.logger.info(f"开始相对旋转 {delta_angle}°")
        
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
        return await self.rotateToAngle(target_angle_normalized, timeout)

    async def rotateLeft(self, angle: float = 90.0, timeout: float = 30.0) -> bool:
        """向左（逆时针）旋转指定角度"""
        self.logger.info(f"向左旋转 {angle}° (逆时针)")
        return await self.rotateByAngle(angle, timeout)

    async def rotateRight(self, angle: float = 90.0, timeout: float = 30.0) -> bool:
        """向右（顺时针）旋转指定角度"""
        self.logger.info(f"向右旋转 {angle}° (顺时针)")
        return await self.rotateByAngle(-angle, timeout)

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
            velocity = util.Velocity.create(angular_z=self.speed)
            self.setVelocity(velocity)
        elif operation == "angularRight":
            # 右角速度+
            velocity = util.Velocity.create(angular_z=-self.speed)
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

                # 旋转到指定角度指令：rotateToAngle:90
                if key.startswith("rotateToAngle:"):
                    try:
                        angle_str = key.split(":")[1]
                        target_angle = float(angle_str)
                        if -180 <= target_angle <= 180:
                            success = await self.rotateToAngle(target_angle)
                            self.logger.info(f"旋转到角度 {target_angle}° {'成功' if success else '失败'}")
                        else:
                            self.logger.warning(f"目标角度超出范围: {target_angle}° (有效范围: -180° ~ 180°)")
                    except (ValueError, IndexError) as e:
                        self.logger.error(f"无效的角度格式: {key}, 错误: {str(e)}")

                # 相对旋转指令：rotateBy:45 或 rotateBy:-30
                elif key.startswith("rotateBy:"):
                    try:
                        angle_str = key.split(":")[1]
                        delta_angle = float(angle_str)
                        if -360 <= delta_angle <= 360:
                            success = await self.rotateByAngle(delta_angle)
                            self.logger.info(f"相对旋转 {delta_angle}° {'成功' if success else '失败'}")
                        else:
                            self.logger.warning(f"旋转角度超出范围: {delta_angle}° (有效范围: -360° ~ 360°)")
                    except (ValueError, IndexError) as e:
                        self.logger.error(f"无效的角度格式: {key}, 错误: {str(e)}")

                # 向左旋转指令：rotateLeft 或 rotateLeft:90
                elif key == "rotateLeft":
                    success = await self.rotateLeft()  # 默认90度
                    self.logger.info(f"向左旋转90° {'成功' if success else '失败'}")
                elif key.startswith("rotateLeft:"):
                    try:
                        angle_str = key.split(":")[1]
                        angle = float(angle_str)
                        if 0 < angle <= 360:
                            success = await self.rotateLeft(angle)
                            self.logger.info(f"向左旋转 {angle}° {'成功' if success else '失败'}")
                        else:
                            self.logger.warning(f"旋转角度超出范围: {angle}° (有效范围: 0° ~ 360°)")
                    except (ValueError, IndexError) as e:
                        self.logger.error(f"无效的角度格式: {key}, 错误: {str(e)}")

                # 向右旋转指令：rotateRight 或 rotateRight:45
                elif key == "rotateRight":
                    success = await self.rotateRight()  # 默认90度
                    self.logger.info(f"向右旋转90° {'成功' if success else '失败'}")
                elif key.startswith("rotateRight:"):
                    try:
                        angle_str = key.split(":")[1]
                        angle = float(angle_str)
                        if 0 < angle <= 360:
                            success = await self.rotateRight(angle)
                            self.logger.info(f"向右旋转 {angle}° {'成功' if success else '失败'}")
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
                            self.logger.info(f"最大旋转速度已设置为: {max_speed} rad/s")
                        else:
                            self.logger.warning(f"最大旋转速度超出范围: {max_speed} (有效范围: 0.1 ~ 3.0)")
                    except (ValueError, IndexError) as e:
                        self.logger.error(f"无效的最大速度格式: {key}, 错误: {str(e)}")

                # 显示当前参数状态
                elif key == "showMotionParams":
                    self.logger.info("=== 当前运动参数 ===")
                    self.logger.info(f"基础速度: {self.speed} m/s")
                    self.logger.info(f"转弯速度: {self.turnSpeed} rad/s")
                    self.logger.info(f"旋转容忍度: {self.rotationTolerance}°")
                    self.logger.info(f"旋转比例系数: {self.rotationKp}")
                    self.logger.info(f"最大旋转速度: {self.maxRotationSpeed} rad/s")
                    self.logger.info(f"速度衰减: {'启用' if self.speedAttenuationEnabled else '禁用'}")
                    if hasattr(self.main, 'imuComponent'):
                        current_yaw = self.main.imuComponent.getYaw()
                        self.logger.info(f"当前偏航角: {current_yaw:.2f}°")
                    self.logger.info("==================")

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(f"运动指令处理异常: {str(e)}")
