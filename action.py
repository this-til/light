import asyncio

from command import CommandEvent
from device import Command, UavBaseStationOperation
from main import Component, ConfigField


class ActionComponent(Component):

    async def awakeInit(self):
        await super().awakeInit()


    async def dispatched(self):
        await self.main.broadcastComponent.playAudio("发现着火目标，已上报服务器")
        await asyncio.sleep(1)
        await self.main.broadcastComponent.playAudio("派遣消防小车前去灭火")
        await asyncio.sleep(1)
        await self.main.broadcastComponent.playAudio("派遣智能无人机协助灭火")
        await asyncio.sleep(1)
        await self.main.deviceComponent.sendCommand(Command(UavBaseStationOperation, "CabinDoorOpen"))
        await asyncio.sleep(1)
        await self.main.deviceComponent.sendCommand(Command(UavBaseStationOperation, "LocalOpen"))

        pass

    async def commandLoop(self):
        queue: asyncio.Queue[CommandEvent] = await self.main.commandComponent.commandEvent.subscribe(
            asyncio.Queue(maxsize=8))

        while True:
            try:
                command: CommandEvent = await queue.get()

                if command.key == "Dispatched":
                    await self.dispatched()

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(f"motionLoop 引发异常: {e}")

    async def instructionLoop(self):
        queue = await self.main.keyComponent.keyEvent.subscribe(asyncio.Queue(maxsize=1))

        while True:

            try:
                key = await queue.get()

                if key == "Dispatched":
                    await self.dispatched()

                if key == "acceptDispatched":
                    await self.main.exclusiveServerReportComponent.setAllowedDispatched(True)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(f"instructionLoop Exception: {str(e)}")
