from __future__ import annotations

import asyncio

import rospy
from geometry_msgs.msg import Twist

from command import CommandEvent
from main import Component, ConfigField


class MotionComponent(Component):
    speed: ConfigField[float] = ConfigField()
    velPub = None  # 添加发布器变量
    msg = None
    speedAttenuationEnabled = True  # 速度衰减开关，默认启用

    async def awakeInit(self):
        await super().awakeInit()
        self.velPub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.msg = Twist()

        asyncio.create_task(self.motionLoop())
        asyncio.create_task(self.speedAttenuationLoop())

    def setVelocity(self, linear_x: float = None, linear_y: float = None, angular_z: float = None):
        """设置速度值的公共函数"""
        if linear_x is not None:
            self.msg.linear.x = linear_x
        if linear_y is not None:
            self.msg.linear.y = linear_y
        if angular_z is not None:
            self.msg.angular.z = angular_z

        self.publishVelocity()

    def publishVelocity(self):
        """发布速度消息的公共函数"""
        if self.velPub:
            self.velPub.publish(self.msg)

    def stopMotion(self):
        """停止所有运动的公共函数"""
        self.setVelocity(linear_x=0, linear_y=0, angular_z=0)

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

    def handleOperation(self, operation: str):
        """处理不同操作类型的公共函数"""
        if operation == "translationAdvance":
            # 向前平移
            self.setVelocity(linear_x=self.speed)
        elif operation == "translationRetreat":
            # 向后平移
            self.setVelocity(linear_x=-self.speed)
        elif operation == "translationLeft":
            # 左平移
            self.setVelocity(linear_y=self.speed)
        elif operation == "translationRight":
            # 右平移
            self.setVelocity(linear_y=-self.speed)
        elif operation == "angularLeft":
            # 左角速度+
            self.setVelocity(angular_z=self.speed)
        elif operation == "angularRight":
            # 右角速度+
            self.setVelocity(angular_z=-self.speed)
        elif operation == "stop":
            # 停止
            self.stopMotion()

    async def motionLoop(self):
        queue: asyncio.Queue[CommandEvent] = await self.main.commandComponent.commandEvent.subscribe(asyncio.Queue(maxsize=8))

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
        return (abs(self.msg.linear.x) > 0.01 or 
                abs(self.msg.linear.y) > 0.01 or 
                abs(self.msg.angular.z) > 0.01)

    def attenuateVelocity(self, attenuation_factor: float = 0.85):
        """速度衰减的公共函数"""
        # 对线性速度进行衰减
        self.msg.linear.x *= attenuation_factor
        self.msg.linear.y *= attenuation_factor
        
        # 对角速度进行衰减
        self.msg.angular.z *= attenuation_factor
        
        # 当速度足够小时，直接设为0
        if abs(self.msg.linear.x) < 0.01:
            self.msg.linear.x = 0.0
        if abs(self.msg.linear.y) < 0.01:
            self.msg.linear.y = 0.0
        if abs(self.msg.angular.z) < 0.01:
            self.msg.angular.z = 0.0

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