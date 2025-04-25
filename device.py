import asyncio
import logging
import uart
import json
import util

uartDataQueue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=16)
logger = logging.getLogger(__name__)

deviceValue =  {}

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
    uart.usarReader.subscribe(uartDataQueue)
    asyncio.create_task(readUartData())
    pass


async def releaseDevice():
    pass
