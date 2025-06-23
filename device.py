#!/usr/bin/python3

import asyncio
import logging
import uart
import json
import util
import copy

import main
from command import CommandEvent

from main import Component, ConfigField


class CommandType:
    target: str
    dataType: str

    def __init__(self, target: str, dataType: str):
        self.target = target
        self.dataType = dataType


SetLightGear: CommandType = CommandType("SetLightGear", "uint")
SetLightSwitch: CommandType = CommandType("LightModeSwitch", "uint")
RollingDoor: CommandType = CommandType("RollingDoor", "uint")


class Command:
    commandType: CommandType
    data: object
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

    commandId: int = 1
    commandIdMap: dict[int, Command] = {}

    dataUpdate: util.Broadcaster[dict] = util.Broadcaster()

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
                # self.logger.debug(f"read data: {strData}")

                frames = await asyncio.get_event_loop().run_in_executor(
                    None, util.splitJsonObjects, strData
                )

                for frame in frames:

                    decoded = json.loads(frame)

                    command_data = decoded.get("Command")
                    if command_data is not None:
                        load = command_data
                        speakId: int = int(load.get("SpeakId", 0))

                        if speakId not in self.commandIdMap:
                            continue

                        command: Command = self.commandIdMap[speakId]
                        command.results = load.get("Execute_Status", "FAILED")
                        command.resultsEvent.set()

                        self.commandIdMap.pop(speakId)

                    data_content = decoded.get("Data")
                    if data_content is not None:
                        self.deviceValue = data_content
                        await self.dataUpdate.publish(copy.deepcopy(self.deviceValue))

                    sensor_data = decoded.get("Sensor")
                    if sensor_data is not None:
                        _sensor = sensor_data

                        await self.main.stateComponent.setStates(
                            {

                                "selfPower": {
                                    "electricity": _sensor.get("Light_Electricity", 0),
                                    "voltage": _sensor.get("Light_Voltage", 0),
                                    "power": _sensor.get("Light_Power", 0)
                                },
                                "wirelessChargingPower": {
                                    "electricity": _sensor.get("Car_Electricity", 0),
                                    "voltage": _sensor.get("Car_Voltage", 0),
                                    "power": _sensor.get("Car_Power", 0)
                                },
                                "uavPower": {
                                    "electricity": _sensor.get("Uav_Electricity", 0),
                                    "voltage": _sensor.get("Uav_Voltage", 0),
                                    "power": _sensor.get("Uav_Power", 0)
                                },
                                "uavBaseStationPower": {
                                    "electricity": _sensor.get("UavBaseStation_Electricity", 0),
                                    "voltage": _sensor.get("UavBaseStation_Voltage", 0),
                                    "power": _sensor.get("UavBaseStation_Power", 0)
                                },

                                "automaticGear": _sensor.get("Light_Mode", 0) == 1,
                                "gear": _sensor.get("Light_Gear", 0),
                                "rollingDoorState": _sensor.get("Rolling_State", "CLOSED")
                            }
                        )

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(f"处理串口数据时发生异常: {str(e)}")

    async def detectionChangeLoop(self):
        queue = await self.main.commandComponent.commandEvent.subscribe(
            asyncio.Queue(maxsize=16)
        )

        while True:
            try:

                event: CommandEvent = await queue.get()
                if event.key == "Device.Gear":
                    await self.sendCommand(Command(SetLightGear, int(event.value)))
                    pass

                if event.key == "Device.Switch":
                    automatic: bool = event.value == "true"
                    await self.sendCommand(Command(SetLightSwitch, 0 if automatic else 1))
                    pass

                if event.key == "Device.RollingDoor":
                    automatic: bool = event.value == "true"
                    await self.sendCommand(Command(RollingDoor, 1 if automatic else 0))

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(f"处理配置数据变更时发生异常: {str(e)}")

    async def sendCommandLoop(self):
        while True:
            try:
                await asyncio.sleep(1)

                if len(self.commandIdMap) == 0:
                    continue

                for commandId, command in list(self.commandIdMap.items()):
                    if command.firstTime == 0:
                        command.firstTime = asyncio.get_event_loop().time()

                    command.lastTime = asyncio.get_event_loop().time()

                    if command.lastTime - command.firstTime > self.commandOutTime:
                        command.resultsEvent.set()
                        logging.error(f"命令超时: {command.commandId}")
                        self.commandIdMap.pop(commandId)
                        continue

                    await self.writeCommand(command)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(f"发送命令时发生异常: {str(e)}")

    async def sendTestLoop(self):
        while True:
            try:
                command = Command(SetLightGear, 1000)
                await self.sendCommand(command)
                await command.wait()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(f"发送测试命令时发生异常: {str(e)}")

    def getDeviceValue(self, key: str):
        return util.getFromJson(key, self.deviceValue)

    async def setDeviceValue(self, key: str, value):
        util.setFromJson(key, value, self.deviceValue)
        await self.dataUpdate.publish(copy.deepcopy(self.deviceValue))

    async def sendCommand(self, command: Command) -> Command:
        self.commandId += 1
        command.commandId = self.commandId
        self.commandIdMap[self.commandId] = command
        await self.writeCommand(command)
        return command

    async def writeCommand(self, command: Command):
        jsonData = command.toJson()
        strData = json.dumps(jsonData)
        self.logger.debug(f"发送命令: {strData}")
        await self.main.uartComponent.writeAsync(strData.encode("utf-8"))
