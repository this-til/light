#!/usr/bin/python3
import logging
import asyncio
import serial_asyncio

import util
import uart
import camera

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

active: bool = True


def exitHandler():
    global active
    active = False
    logger.debug("Exiting...")

async def main():

    try:

        await uart.initUart()
        await camera.initCamera()

        while active:
            await asyncio.sleep(1)

    finally:

        await util.gracefulShutdown()

        await uart.releaseUart()
        await camera.releaseCamera()
        
        exit(0)
        pass


if __name__ == "__main__":
    asyncio.run(main())
