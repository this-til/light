import asyncio
import logging
import json
import util
import copy  # 添加拷贝模块

import paho.mqtt.client as mqtt

from main import Component, ConfigField

class StateComponent(Component):
    stateValue: dict[str, object] = {}
    stateChange: util.Broadcaster[dict[str, object]] = util.Broadcaster()

    async def setStates(self, state : dict[str, object]) -> None:
        self.stateValue = state
        await self.stateChange.publish(copy.deepcopy(self.stateValue))

    async def setState(self, key : str, value : str) -> None:
        util.setFromJson(key, value, self.stateValue)
        await self.stateChange.publish(copy.deepcopy(self.stateValue))
