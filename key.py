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


class KeyEvent:
    key: str = 0
    special: bool = False


class KeyComponent(Component):
    keyEvent: util.Broadcaster[KeyEvent] = util.Broadcaster()

    async def init(self):
        await super().init()

        asyncio.create_task(self.keyboardListener())
        asyncio.create_task(self.processKeyEventLogLoop())

    async def keyboardListener(self):
        while True:
            # 使用select检查输入是否有数据
            # rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
            rlist = asyncio.get_event_loop().run_in_executor(
                None, select.select, [sys.stdin], [], [], 0.1
            )

            if rlist:
                key = sys.stdin.read(1)

                event = KeyEvent()
                event.key = key
                event.special = len(key) > 1

                await self.keyEvent.publish(event)

    async def processKeyEventLogLoop(self):
        queue = await self.keyEvent.subscribe(asyncio.Queue(maxsize=16))

        while True:
            # 从队列获取事件
            event = await queue.get()

            self.logger.info(f"key : {event.key}")
