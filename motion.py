from __future__ import annotations

import asyncio

import rospy
from geometry_msgs.msg import Twist

from command import CommandEvent
from main import Component, ConfigField
import util


class MotionComponent(Component):
    speed: ConfigField[float] = ConfigField()
    turnSpeed: ConfigField[float] = ConfigField()

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
    
    async def motionTimeWithComponents(self, linear: util.V3 = None, angular: util.V3 = None, time: float = 1):
        """使用向量组件运动指定时间"""
        velocity = util.Velocity(linear, angular)
        await self.motionTime(velocity=velocity, time=time)
