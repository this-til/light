#!/usr/bin/python3False
import logging
import asyncio
import server
import device
import configure
import signal

import util
import uart
import camera
import detection

logging.basicConfig(
    level=logging.DEBUG, format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

active: bool = True


async def main():

    loop = asyncio.get_running_loop()
    active = True  # 主循环控制变量

    # 定义信号处理器
    def _signal_handler():
        nonlocal active
        active = False
        logger.debug("捕获到 Ctrl+C，开始优雅关闭...")

    # 注册 SIGINT 处理器
    loop.add_signal_handler(signal.SIGINT, _signal_handler)

    try:

        await configure.initConfigure()
        await uart.initUart()
        await camera.initCamera()
        # await server.initServer()
        await device.initDevice()
        await detection.initDetection()

        await server.runServer()

        # while active:
        #    await asyncio.sleep(1)
    finally:

        await util.gracefulShutdown()

        await detection.releaseDetection()
        await device.releaseDevice()
        await server.releaseServer()
        await camera.releaseCamera()
        await uart.releaseUart()
        await configure.releaseConfigure()

        pass


if __name__ == "__main__":
    asyncio.run(main())
