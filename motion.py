import asyncio

from command import CommandEvent
from main import Component, ConfigField


class MotionComponent(Component):
    operationCarMap: dict[str, object] = {
        "translationAdvance": None,
        "translationLeft": None,
        "translationRetreat": None,
        "translationRight": None,
        "angularLeft": None,
        "angularRight": None,
        "stop": None,
    }

    async def awakeInit(self):
        return await super().awakeInit()

    async def motionLoop(self):

        queue: asyncio.Queue[CommandEvent] = self.main.commandComponent.commandEvent.subscribe(asyncio.Queue(maxsize=8))

        while True:
            try:
                command: CommandEvent = await queue.get()

                if command.key == "Operation":

                    if not command.value in self.operationCarMap:
                        continue

                    com = self.operationCarMap[command.value]  # TODO


            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(f"motionLoop 引发异常: {e}")

        pass
