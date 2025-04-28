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
            "Humidity": 46,
            "Temperature": 24,
            "PM10": 12,
            "PM2.5": 9,
            "Illuminance": 183,
        },
        "Wind_Speed": {"Wind_Speed": 0, "Wind_Direction": 315},
        "Distance_Front": {"Distance": 100},
        "Distance_Rear": {"Distance": 200},
        "Distance_Left": {"Distance": 300},
        "Distance_Right": {"Distance": 400},
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
