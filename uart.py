#!/usr/bin/python3

import logging
import asyncio
import serial_asyncio

logger = logging.getLogger(__name__)

loop = asyncio.get_event_loop()
loop.set_debug(True)

reader: asyncio.StreamReader = None
writer: asyncio.StreamWriter = None


async def asyncReadSerialLoop():
    """
    异步循环读取
    """
    while True:
        try:
            data = await reader.read(1024)
            # TODO
            await asyncio.sleep(0.1)  # 休眠100ms
        except Exception as e:
            logger.exception(f"Error reading from serial: {str(e)}")
            return


async def initUart():

    global reader
    global writer

    reader, writer = await serial_asyncio.open_serial_connection(
        url="/dev/ttyS9",
        baudrate=115200,
        bytesize=serial_asyncio.serial.EIGHTBITS,
        parity=serial_asyncio.serial.PARITY_NONE,
        stopbits=serial_asyncio.serial.STOPBITS_ONE,
    )

    asyncio.create_task(asyncReadSerialLoop())

    pass


async def releaseUart():
    """
    释放串口
    """
    if writer and not writer.is_closing():
        writer.close()
        await writer.wait_closed()

    pass
