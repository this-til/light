#!/usr/bin/python3

import asyncio
import logging
import uart
import json
import util

import main
from main import Component, ConfigField

class DeviceComponent(Component):

    deviceValue = {
        "Modbus": {
            "Weather": {
                "Humidity": 49,
                "Temperature": 24,
                "PM10": 38,
                "PM2.5": 29,
                "Illuminance": 212,
                "Failed_Number": 1,
                "State": "Normal",
            },
            "Wind_Speed": {
                "Wind_Speed": 0,
                "Wind_Direction": 337.5,
                "Failed_Number": 0,
                "State": "Normal",
            },
            "Distance_Front": {"Distance": 345, "Failed_Number": 0, "State": "Normal"},
            "Distance_Rear": {"Distance": 200, "Failed_Number": 10, "State": "Error"},
            "Distance_Left": {"Distance": 300, "Failed_Number": 10, "State": "Error"},
            "Distance_Right": {"Distance": 400, "Failed_Number": 10, "State": "Error"},
        }
    }

    uartDataQueue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=16)

    async def init(self):
        await super().init()
        await main.uartComponent.usarReader.subscribe(self.uartDataQueue)
        asyncio.create_task(self.readUartDataLoop())

    async def readUartDataLoop(self):
        while True:

            try:
                data: bytes = await self.uartDataQueue.get()
                decoded = data.decode("utf-8").strip()
                util.jsonDeepMerge(self.deviceValue, decoded)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(f"处理串口数据时发生异常: {str(e)}")

    def getDeviceValue(self, key: str):
        return util.getFromJson(key, self.deviceValue)

    def setDeviceValue(self, key: str, value):
        util.setFromJson(key, value, self.deviceValue)
