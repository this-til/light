#!/usr/bin/python3False
import logging
import asyncio
import configure
import signal

import util
import uart
import camera
import orbbec_camera
import detection
import audio
import server
import device
import mqtt
import hkws_sdk

from util import T
from typing import Generator

logging.basicConfig(
    level=logging.DEBUG, format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class ConfigField(Generator[T]):

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
    configFields: list[str] = []

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    async def awakeInit(self):
        if not isinstance(self, configure.ConfigureComponent):
            for field in self.configFields:
                setattr(
                    type(self),
                    field,
                    configureComponent.getConfigure(
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


configureComponent = configure.ConfigureComponent()
uartComponent = uart.UartComponent()
orbbecCameraComponent = orbbec_camera.OrbbecCameraComponent()
deviceComponent = device.DeviceComponent()
mqttComponent = mqtt.MqttComponent()
detectionComponent = detection.DetectionComponent()
cameraComponent = camera.CameraComponent()
hkwsSdkComponent = hkws_sdk.HCNetSdkComponent()

components: list[Component] = [
    configureComponent,
    uartComponent,
    orbbecCameraComponent,
    deviceComponent,
    mqttComponent,
    detectionComponent,
    hkwsSdkComponent,
    cameraComponent,
]


async def main():

    components.sort(key=lambda component: component.getPriority(), reverse=True)
    logger.debug(f"组件排序完成, 组件数量: {len(components)}")

    for component in components:
        try:
            await component.awakeInit()
        except Exception as e:
            logger.exception(
                f"{component.__class__.__name__} awakeInit() 失败, 错误: {e}"
            )
            continue

    logger.debug("组件awakeInit完成")

    for component in components:
        try:
            await component.init()
        except Exception as e:
            logger.exception(f"{component.__class__.__name__} init() 失败, 错误: {e}")
            continue

    logger.debug("组件init完成")

    for component in components:
        try:
            await component.initBack()
        except Exception as e:
            logger.exception(
                f"{component.__class__.__name__} initBack() 失败, 错误: {e}"
            )
            continue

    logger.debug("组件initBack完成")

    try:
        await server.runServer()
    finally:
        await util.gracefulShutdown()

        for component in components:
            try:
                await component.release()
            except Exception as e:
                logger.exception(
                    f"{component.__class__.__name__} release() 失败, 错误: {e}"
                )
                continue

        logger.debug("组件release完成")

        for component in components:
            try:
                await component.exitBack()
            except Exception as e:
                logger.exception(
                    f"{component.__class__.__name__} exitBack() 失败, 错误: {e}"
                )
                continue

        logger.debug("组件exitBack完成")


if __name__ == "__main__":
    asyncio.run(main())
