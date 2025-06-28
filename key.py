import sys
import select
import tty
import termios
import asyncio
import signal
from collections import deque
from typing import AsyncGenerator, Callable, Optional

import util
from main import Component, ConfigField


class KeyComponent(Component):
    keyEvent: util.Broadcaster[str] = util.Broadcaster()

    async def init(self):
        await super().init()

        asyncio.create_task(self.keyboardListener())

    async def keyboardListener(self):
        while True:
            # 使用select检查输入是否有数据
            # rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
            try:
                # rlist = await asyncio.get_event_loop().run_in_executor(
                #    None, select.select, [sys.stdin], [], [], 0.1
                # )

                # if rlist:
                # key = sys.stdin.read(1)
                key = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )

                key = key.rstrip("\n")
                self.logger.info(f"key : {key}")

                await self.keyEvent.publish(key)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                if not self.main.run:
                    raise
                self.logger.exception(f"keyboardListener exception: {str(e)}")
                await asyncio.sleep(5)
