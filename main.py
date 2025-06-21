#!/usr/bin/python3False
from __future__ import annotations

import logging
import logging.config
import asyncio
import rospy
import cv2

import util
from typing import Generic, TypeVar

""" logging.basicConfig(
    level=logging.DEBUG, format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
) """

logger = logging.getLogger(__name__)

T = TypeVar("T")  # 定义泛型类型

class ROSLogHandler(logging.Handler):
    """将 Python logging 重定向到 ROS 日志系统"""
    
    def emit(self, record):
        ## 映射日志级别到 ROS 对应函数
        #log_functions = {
        #    logging.DEBUG: rospy.logdebug,
        #    logging.INFO: rospy.loginfo,
        #    logging.WARNING: rospy.logwarn,
        #    logging.ERROR: rospy.logerr,
        #    logging.CRITICAL: rospy.logfatal
        #}
        #
        ## 获取匹配的 ROS 日志函数（默认使用 logerr）
        #log_func = log_functions.get(record.levelno, rospy.logerr)
        #
        ## 格式化日志消息并发送到 ROS
        #message = self.format(record)
        #log_func(message)
        
        try:
            print(self.format(record))
        except Exception as e:
            pass
        

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
    
        rospy.init_node("car_python")
        
        handler = ROSLogHandler()
        
        formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)

        log_level = rospy.get_param("~log_level", "DEBUG")
        root_logger.setLevel(log_level)
        
    
        from configure import ConfigureComponent
        from orbbec_camera import OrbbecCameraComponent
        from detection import DetectionComponent
        from report import ExclusiveServerReportComponent
        from state import StateComponent
        from command import CommandComponent
        from laser_radar import LaserRadarComponent
        from action import ActionComponent
        from motion import MotionComponent
        from broadcast import BroadcastComponent

        self.configureComponent = ConfigureComponent()
        self.orbbecCameraComponent = OrbbecCameraComponent()
        self.exclusiveServerReportComponent = ExclusiveServerReportComponent()
        self.detectionComponent = DetectionComponent()
        self.stateComponent = StateComponent()
        self.commandComponent = CommandComponent()
        self.laserRadarComponent = LaserRadarComponent()
        self.actionComponent = ActionComponent()
        self.motionComponent = MotionComponent()
        self.broadcastComponent = BroadcastComponent()

        self.components.append(self.configureComponent)
        self.components.append(self.orbbecCameraComponent)
        self.components.append(self.exclusiveServerReportComponent)
        self.components.append(self.detectionComponent)
        self.components.append(self.stateComponent)
        self.components.append(self.commandComponent)
        self.components.append(self.laserRadarComponent)
        self.components.append(self.actionComponent)
        self.components.append(self.motionComponent)
        self.components.append(self.broadcastComponent)

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
    
