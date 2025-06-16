import asyncio
import logging
import json
import util

import paho.mqtt.client as mqtt

from main import Component, ConfigField


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
