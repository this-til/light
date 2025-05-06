#!/usr/bin/python3

import asyncio
import logging
import uart
import json
import util

uartDataQueue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=16)
logger = logging.getLogger(__name__)

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


async def readUartData():
    while True:

        try:
            data: bytes = await uartDataQueue.get()
            decoded = data.decode("utf-8").strip()
            util.jsonDeepMerge(deviceValue, decoded)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception(f"处理串口数据时发生异常: {str(e)}")


async def initDevice():
    await uart.usarReader.subscribe(uartDataQueue)
    asyncio.create_task(readUartData())
    pass


async def releaseDevice():
    pass


def getDeviceValue(key: str):
    return util.getFromJson(key, deviceValue)


def setDeviceValue(key: str, value):
    util.setFromJson(key, value, deviceValue)
