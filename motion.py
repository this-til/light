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

    async def awakeInit(self):
        await super().awakeInit()
        self.velPub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.msg = Twist()

        asyncio.create_task(self.motionLoop())
        asyncio.create_task(self.speedAttenuationLoop())

    async def motionLoop(self):

        queue: asyncio.Queue[CommandEvent] = self.main.commandComponent.commandEvent.subscribe(asyncio.Queue(maxsize=8))
        msg = self.msg

        while True:
            try:
                command: CommandEvent = await queue.get()

                if command.key == "Operation":

                    if command.value == "translationAdvance":
                        # 向前平移
                        self.msg.linear.x = self.speed

                    elif command.value == "translationRetreat":
                        # 向后平移
                        self.msg.linear.x = self.speed

                    elif command.value == "translationLeft":
                        # 左平移
                        self.msg.linear.y = self.speed

                    elif command.value == "translationRight":
                        # 右平移
                        self.msg.linear.y = self.speed

                    elif command.value == "angularLeft":
                        # 左角速度+
                        self.msg.angular.z = self.speed

                    elif command.value == "angularRight":
                        # 右角速度+
                        self.msg.angular.z = self.speed

                    elif command.value == "stop":
                        # 停止
                        self.msg = Twist()

                    # 发布速度消息
                    if self.velPub:
                        self.velPub.publish(msg)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(f"motionLoop 引发异常: {e}")

    async def speedAttenuationLoop(self):
        while True:
            try:
                # 每0.1秒检查一次速度衰减
                await asyncio.sleep(0.1)
                
                # 衰减因子，1秒内衰减到0，每0.1秒衰减10%
                attenuation_factor = 0.9
                
                # 检查是否有速度需要衰减
                has_velocity = (abs(self.msg.linear.x) > 0.01 or 
                              abs(self.msg.linear.y) > 0.01 or 
                              abs(self.msg.angular.z) > 0.01)
                
                if has_velocity:
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
                    
                    # 发布衰减后的速度消息
                    if self.velPub:
                        self.velPub.publish(self.msg)
                        
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(f"speedAttenuationLoop 引发异常: {e}")