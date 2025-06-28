#!/usr/bin/python3False
from __future__ import annotations

import sys
import platform
import os
import datetime

print("=" * 40, "Environment Information", "=" * 40)

try:
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(f"OS Platform: {platform.platform()}")
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Processor: {platform.processor()}")
    print(f"Current Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working Directory: {os.getcwd()}")
    print(f"Username: {os.getlogin()}")
    print(f"Machine Name: {platform.node()}")
except Exception as e:
    print(e)

print("=" * 100)


import logging
import logging.config
import asyncio
import rospy
import cv2

import util
from typing import Generic, TypeVar

logging.basicConfig(
    level=logging.DEBUG, format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

T = TypeVar("T")  # 定义泛型类型


class ConfigField(Generic[T]):
    default: T = None  # type: ignore

    def __init__(self, default: T | None = None):
        self.default = default  # type: ignore

    def __get__(self, instance, owner):
        return self.default

    def __set__(self, instance, value):
        self.default = value


class ComponentMeta(type):
    def __new__(cls, name, bases, dct):
        fields = [key for key, value in dct.items() if isinstance(value, ConfigField)]
        obj = super().__new__(cls, name, bases, dct)
        obj.configFields = fields  # type: ignore
        return obj


class Component(metaclass=ComponentMeta):
    logger: logging.Logger
    main: "Mian" = None  # type: ignore
    configFields: list[str] = []

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    async def awakeInit(self):
        for field in self.configFields:
            setattr(
                type(self),
                field,
                self.main.configureComponent.getConfigure(
                    f"{self.__class__.__name__}.{field}"
                ),
            )
        pass

    async def init(self):
        pass

    async def initBack(self):
        pass

    async def initEnd(self):
        pass

    async def release(self):
        pass

    async def exitBack(self):
        pass

    def getPriority(self) -> int:
        return 0

    pass


class Mian:
    components: list[Component] = []

    run: bool = True

    def __init__(self):

        pass

    async def main(self):
        from ros_access import RosAccessComponent
        from configure import ConfigureComponent
        from orbbec_camera import OrbbecCameraComponent
        from detection import DetectionComponent
        from report import ExclusiveServerReportComponent
        from state import StateComponent
        from command import CommandComponent
        from action import ActionComponent
        from motion import MotionComponent
        from broadcast import BroadcastComponent
        from key import KeyComponent
        from imu import ImuComponent
        
        self.rosAccessComponent = RosAccessComponent()
        self.configureComponent = ConfigureComponent()
        self.orbbecCameraComponent = OrbbecCameraComponent()
        self.exclusiveServerReportComponent = ExclusiveServerReportComponent()
        self.detectionComponent = DetectionComponent()
        self.stateComponent = StateComponent()
        self.commandComponent = CommandComponent()
        self.actionComponent = ActionComponent()
        self.motionComponent = MotionComponent()
        self.broadcastComponent = BroadcastComponent()
        self.keyComponent = KeyComponent()
        self.imuComponent = ImuComponent()
        

        self.components.append(self.rosAccessComponent)
        self.components.append(self.configureComponent)
        self.components.append(self.orbbecCameraComponent)
        self.components.append(self.exclusiveServerReportComponent)
        self.components.append(self.detectionComponent)
        self.components.append(self.stateComponent)
        self.components.append(self.commandComponent)
        self.components.append(self.actionComponent)
        self.components.append(self.motionComponent)
        self.components.append(self.broadcastComponent)
        self.components.append(self.keyComponent)
        self.components.append(self.imuComponent)

        for component in self.components:
            component.main = self  # type: ignore

        _components = self.components.copy()
        _components.sort(key=lambda component: component.getPriority(), reverse=True)

        logger.debug(f"组件排序完成, 组件数量: {len(_components)}")

        for component in _components:
            try:
                logger.debug(f"组件: {component.__class__.__name__} awakeInit()")
                await component.awakeInit()
            except Exception as e:
                logger.exception(
                    f"{component.__class__.__name__} awakeInit() 失败, 错误: {e}"
                )
                continue

        await asyncio.sleep(0.1)

        logger.debug("组件awakeInit完成")

        for component in _components:
            try:
                logger.debug(f"组件: {component.__class__.__name__} init()")
                await component.init()
            except Exception as e:
                logger.exception(
                    f"{component.__class__.__name__} init() 失败, 错误: {e}"
                )
                continue

        await asyncio.sleep(0.1)

        logger.debug("组件init完成")

        for component in _components:
            try:
                logger.debug(f"组件: {component.__class__.__name__} initBack()")
                await component.initBack()
            except Exception as e:
                logger.exception(
                    f"{component.__class__.__name__} initBack() 失败, 错误: {e}"
                )
                continue

        await asyncio.sleep(0.1)

        logger.debug("组件initBack完成")

        try:
            #while self.run:
            #    await asyncio.sleep(5)
            while not rospy.is_shutdown():
                await asyncio.sleep(5)
        finally:
            self.run = False
            
            cv2.destroyAllWindows()
            
            await util.gracefulShutdown()

            for component in _components:
                try:
                    logger.debug(f"组件: {component.__class__.__name__} release()")
                    await component.release()
                except Exception as e:
                    logger.exception(
                        f"{component.__class__.__name__} release() 失败, 错误: {e}"
                    )
                    continue

            await asyncio.sleep(0.1)

            logger.debug("组件release完成")

            for component in _components:
                try:
                    logger.debug(f"组件: {component.__class__.__name__} exitBack()")
                    await component.exitBack()
                except Exception as e:
                    logger.exception(
                        f"{component.__class__.__name__} exitBack() 失败, 错误: {e}"
                    )
                    continue

            await asyncio.sleep(0.1)

            logger.debug("组件exitBack完成")



if __name__ == "__main__":
    asyncio.run(Mian().main())
    
