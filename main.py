#!/usr/bin/python3False
import logging
import asyncio
import server
import device
from flask import Flask


import util
import uart
import camera

app = Flask(__name__)

logging.basicConfig(
    level=logging.DEBUG, format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
)
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
        await server.initServer()
        await device.initDevice()

        while active:
            await asyncio.sleep(1)

    finally:

        await util.gracefulShutdown()

        await uart.releaseUart()
        await camera.releaseCamera()
        await server.releaseServer()
        await device.releaseDevice()

        exit(0)
        pass


if __name__ == "__main__":
    asyncio.run(main())
