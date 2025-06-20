from __future__ import annotations

import util
from main import Component


class CommandEvent:
    key: str = None
    value: str = None

    def __init__(self, key: str, value: str):
        self.key = key
        self.value = value

    pass


class CommandComponent(Component):
    commandEvent: util.Broadcaster[CommandEvent] = util.Broadcaster()

    async def onCommand(self, key: str, value: str) -> None:
        await self.commandEvent.publish(CommandEvent(key, value))

    pass
