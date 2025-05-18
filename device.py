#!/usr/bin/python3

import asyncio
import logging
import uart
import json
import util

import main

from main import Component, ConfigField


class CommandType:

    target: str
    dataType: str

    def __init__(self, target: str, dataType: str):
        self.target = target
        self.dataType = dataType


SetLightGear: CommandType = CommandType("SetLightGear", "uint")
SetLightSwitch: CommandType = CommandType("SetLightSwitch", "uint")


class Command:

    commandType: CommandType
    dtat: object
    commandId: int

    resultsEvent: asyncio.Event = asyncio.Event()
    results: object | None = None

    firstTime: float = 0
    lastTime: float = 0

    def __init__(self, commandType: CommandType, data: object):
        self.commandType = commandType
        self.data = data
        self.resultsEvent.clear()

    def toJson(self) -> dict:
        return {
            "Target": self.commandType.target,
            "Type": self.commandType.dataType,
            "Data": self.data,
            "SpeakId": self.commandId,
        }

    async def wait(self) -> object | None:
        await self.resultsEvent.wait()
        return self.results


class DeviceComponent(Component):

    commandOutTime: ConfigField[float] = ConfigField()

    deviceValue = {}

    commandId: int = 0
    commandIdMap: dict[int, Command] = {}

    dataUpdate: util.EventBroadcaster = util.EventBroadcaster()

    async def init(self):
        await super().init()

        asyncio.create_task(self.readUartDataLoop())
        asyncio.create_task(self.detectionChangeLoop())
        asyncio.create_task(self.sendCommandLoop())
        # asyncio.create_task(self.sendTestLoop())

    async def readUartDataLoop(self):
        uartDataQueue = await self.main.uartComponent.usarReader.subscribe(
            asyncio.Queue(maxsize=16)
        )

        while True:

            try:
                data: bytes = await uartDataQueue.get()
                strData = data.decode("utf-8").strip()
                self.logger.debug(f"read dtat: {strData}")
                decoded = json.loads(strData)

                if "Command" in decoded:
                    load = decoded["Command"]
                    speakId: int = int(load["SpeakId"])

                    if speakId not in self.commandIdMap:
                        continue

                    command: Command = self.commandIdMap[speakId]
                    command.results = load["Execute_Status"]
                    command.resultsEvent.set()

                    self.commandIdMap.pop(speakId)

                if "Data" in decoded:
                    util.jsonDeepMerge(self.deviceValue, decoded["Data"])
                    await self.dataUpdate.setEvent()

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(f"处理串口数据时发生异常: {str(e)}")

    async def detectionChangeLoop(self):
        queue = await self.main.configureComponent.configureChange.subscribe(
            asyncio.Queue(maxsize=16)
        )

        while True:
            try:

                event = await queue.get()
                if event.key == "Device.Gear":
                    self.sendCommand(Command(SetLightGear, event.value))
                    pass

                if event.key == "Device.Switch":
                    self.sendCommand(Command(SetLightSwitch, event.value))
                    pass

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(f"处理配置数据变更时发生异常: {str(e)}")

    async def sendCommandLoop(self):
        while True:

            await asyncio.sleep(1)

            try:
                for commandId, command in self.commandIdMap.items():
                    if command.firstTime == 0:
                        command.firstTime = asyncio.get_event_loop().time()

                    command.lastTime = asyncio.get_event_loop().time()

                    if command.lastTime - command.firstTime > self.commandOutTime:
                        command.resultsEvent.set()
                        logging.error(f"命令超时: {command.commandId}")
                        del self.commandIdMap[commandId]
                        continue

                    jsonData = command.toJson()
                    strData = json.dumps(jsonData)
                    self.logger.debug(f"发送命令: {strData}")
                    self.main.uartComponent.write(strData.encode("utf-8"))

                    await asyncio.sleep(1)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(f"发送命令时发生异常: {str(e)}")

    async def sendTestLoop(self):
        while True:
            try:
                command = Command(SetLightGear, 1000)
                self.sendCommand(command)
                await command.wait()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(f"发送测试命令时发生异常: {str(e)}")

    def getDeviceValue(self, key: str):
        return util.getFromJson(key, self.deviceValue)

    def setDeviceValue(self, key: str, value):
        util.setFromJson(key, value, self.deviceValue)
        self.dataUpdate.setEvent_nowait()

    def sendCommand(self, command: Command) -> Command:
        self.commandId += 1
        command.commandId = self.commandId
        self.commandIdMap[self.commandId] = command
        return command
