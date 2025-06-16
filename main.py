#!/usr/bin/python3False
import logging
import logging.config
import asyncio

import util

from typing import Generic, TypeVar

logging.basicConfig(
   level=logging.DEBUG, format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
)

logging.config.dictConfig(
    {
        "version": 1,
        "loggers": {
            "websockets.client": {  # 专门针对gql库
                "level": "WARN",  # 设置日志等级
                "handlers": ["console"],
                "propagate": False,  # 阻止传播到根记录器
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "simple",
                "level": "DEBUG",
            }
        },
        "formatters": {
            "simple": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
        },
    }
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

    def __init__(self):

        # self.configureComponent = None
        # self.uartComponent = None
        # self.orbbecCameraComponent = None
        # self.deviceComponent = None
        # self.mqttReportComponent = None
        # self.exclusiveServerReportComponent = None
        # self.detectionComponent = None
        # self.cameraComponent = None
        # self.hkwsSdkComponent = None
        # self.audioComponent = None
        # self.serverComponent = None
        # self.microphoneComponent = None

        pass

    async def main(self):

        from configure import ConfigureComponent
        from uart import UartComponent
        from camera import CameraComponent
        from orbbec_camera import OrbbecCameraComponent
        from detection import DetectionComponent
        from audio import AudioComponent
        from device import DeviceComponent
        from report import MqttReportComponent
        from report import ExclusiveServerReportComponent
        from hkws_sdk import HCNetSdkComponent
        from server import ServerComponent
        from microphone import MicrophoneComponent

        self.configureComponent = ConfigureComponent()
        self.uartComponent = UartComponent()
        self.orbbecCameraComponent = OrbbecCameraComponent()
        self.deviceComponent = DeviceComponent()
        self.mqttReportComponent = MqttReportComponent()
        self.exclusiveServerReportComponent = ExclusiveServerReportComponent()
        self.detectionComponent = DetectionComponent()
        self.cameraComponent = CameraComponent()
        self.hkwsSdkComponent = HCNetSdkComponent()
        self.audioComponent = AudioComponent()
        self.serverComponent = ServerComponent()
        self.microphoneComponent = MicrophoneComponent()

        self.components.append(self.configureComponent)
        self.components.append(self.uartComponent)
        self.components.append(self.orbbecCameraComponent)
        self.components.append(self.deviceComponent)
        self.components.append(self.mqttReportComponent)
        self.components.append(self.exclusiveServerReportComponent)
        self.components.append(self.detectionComponent)
        self.components.append(self.hkwsSdkComponent)
        self.components.append(self.cameraComponent)
        self.components.append(self.audioComponent)
        self.components.append(self.serverComponent)
        self.components.append(self.microphoneComponent)

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
            await self.serverComponent.runServer()
        finally:
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

        pass


if __name__ == "__main__":
    asyncio.run(Mian().main())
