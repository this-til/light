import asyncio
import curses

from main import Component, ConfigField

class KeyComponent(Component):

    async def awakeInit(self):
        await super().awakeInit()
        asyncio.create_task(self.readLoop())

    async def readLoop(self):
        while True:
            try:
                pass
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(f"KeyComponent.readLoop: {e}")
