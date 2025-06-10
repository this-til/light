import asyncio
import logging
import json
import util
import backoff
from io import BytesIO

import cv2
import paho.mqtt.client as mqtt
from gql.client import AsyncClientSession

import detection
import main
from typing import cast
from gql import gql, Client
from gql.transport.exceptions import TransportError
from gql.transport.websockets import WebsocketsTransport
from gql.transport.aiohttp import AIOHTTPTransport
from websockets import Subprotocol, WebSocketException

from main import Component, ConfigField
from util import Box

import aiohttp

logger = logging.getLogger(__name__)


class LightState:
    enableWirelessCharging: bool
    wirelessChargingPower: float


class MqttReportComponent(Component):
    enable: ConfigField[bool] = ConfigField()

    receiveTopic: ConfigField[str] = ConfigField()
    updataTopic: ConfigField[str] = ConfigField()
    clientId: ConfigField[str] = ConfigField()
    ip: ConfigField[str] = ConfigField()
    port: ConfigField[int] = ConfigField()
    userName: ConfigField[str] = ConfigField()
    password: ConfigField[str] = ConfigField()

    client: mqtt.Client = None  # type: ignore

    async def init(self):
        await super().init()

        if self.enable:
            self.client = mqtt.Client(client_id=self.clientId)

            self.client.subscribe(self.receiveTopic)
            self.client.subscribe(self.updataTopic)

            self.client.on_connect = self.onConnect
            self.client.on_message = self.onMessage

            self.client.username_pw_set(self.userName, self.password)
            self.client.connect(self.ip, self.port)
            self.client.loop_start()  # 使用非阻塞循环

        # asyncio.create_task(self.mqttReportLoop())

    async def mqttReportLoop(self):

        queue = await self.main.deviceComponent.dataUpdate.subscribe(asyncio.Queue(maxsize=8))

        while True:

            try:
                data = await queue.get()
                modbus = data["Modbus"]
                weather = modbus["Weather"]

                windSpeed = modbus["Wind_Speed"]

                self.client.publish(
                    self.updataTopic,
                    json.dumps(
                        {
                            "湿度": weather["Humidity"],
                            "温度": weather["Temperature"],
                            "PM10": weather["PM10"],
                            "PM2.5": weather["PM2.5"],
                            "光照": weather["Illuminance"],
                            "风速": windSpeed["Wind_Speed"],
                            "风向": windSpeed["Wind_Direction"],
                        }
                    ),
                )

                await asyncio.sleep(5)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(f"发布数据时发生异常: {str(e)}")

    def onConnect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.debug("成功连接至MQTT服务器!")
        else:
            logger.debug(f"连接失败，错误码: {rc}")

    def onMessage(self, client, userdata, msg):
        logger.debug(f"收到消息: Topic={msg.topic}, Payload={msg.payload.decode()}")


class ExclusiveServerReportComponent(Component):
    enable: ConfigField[bool] = ConfigField()

    username: ConfigField[str] = ConfigField()
    password: ConfigField[str] = ConfigField()
    localName: ConfigField[str] = ConfigField()

    webSocketUrl: ConfigField[str] = ConfigField()
    httpUrl: ConfigField[str] = ConfigField()
    timeout: ConfigField[int] = ConfigField()
    pingInterval: ConfigField[int] = ConfigField()
    subprotocol: ConfigField[str] = ConfigField()

    dataUpdateQueue: asyncio.Queue[dict] = asyncio.Queue(maxsize=8)
    stateUpdateQueue: asyncio.Queue[LightState] = asyncio.Queue(
        maxsize=8
    )  # TODO subscribe stateUpdateQueue
    identifyKeyframeQueue: asyncio.Queue[detection.Result] = asyncio.Queue(maxsize=8)

    async def init(self):
        if self.enable:
            await self.main.deviceComponent.dataUpdate.subscribe(self.dataUpdateQueue)
            await self.main.cameraComponent.identifyKeyframe.subscribe(
                self.identifyKeyframeQueue
            )

            asyncio.create_task(self.webSocketTransportLoop())
            asyncio.create_task(self.detectionReportLoop())
        pass

    def establishLink(self) -> WebsocketsTransport:

        return WebsocketsTransport(
            url=self.webSocketUrl,
            init_payload={
                "username": self.username,
                "password": self.password,
                "linkType": "DEVICE_WEBSOCKET",
                "deviceType": "LIGHT",
                "deviceName": self.localName,
            },
            keep_alive_timeout=self.timeout,
            ping_interval=self.pingInterval,
            subprotocols=[cast(Subprotocol, self.subprotocol)],
        )

    # def publishTask(self, tasks: list[asyncio.Task], session: AsyncClientSession):

    #    task = tasks[0]
    #    if task is None or task.done():
    #        tasks[0] = asyncio.create_task(self.sensorReportLoop(session))
    #    task = tasks[1]
    #    if task is None or task.done():
    #        tasks[1] = asyncio.create_task(self.stateReportLoop(session))
    #    task = tasks[2]
    #    if task is None or task.done():
    #        tasks[2] = asyncio.create_task(self.configurationDistributionLoop(session))

    #    pass

    async def webSocketTransportLoop(self):
        ws = self.establishLink()

        client = Client(
            transport=ws,
            fetch_schema_from_transport=False
        )

        try:


            session = await client.connect_async(
                True,
                retry_execute=False,
                retry_connect=backoff.on_exception(
                    backoff.constant,
                    Exception,
                    interval=5,
                )
            )

            taskList = [
                asyncio.create_task(self.sensorReportLoop(session)),
                asyncio.create_task(self.stateReportLoop(session)),
                asyncio.create_task(self.configurationDistributionLoop(session)),
            ]

            done, pending = await asyncio.wait(
                taskList, return_when=asyncio.FIRST_EXCEPTION
            )

            await util.gracefulShutdown(taskList)

        except asyncio.CancelledError:
            raise
        finally:
            await client.close_async()

    #    async def webSocketTransportLoop(self):
    #        ws = self.establishLink()
    #
    #        while True:
    #
    #            try:
    #                async with Client(
    #                        transport=ws, fetch_schema_from_transport=False
    #                ) as session:
    #
    #                    taskList = [
    #                        asyncio.create_task(self.sensorReportLoop(session)),
    #                        asyncio.create_task(self.stateReportLoop(session)),
    #                        asyncio.create_task(self.configurationDistributionLoop(session)),
    #                    ]
    #
    #                    done, pending = await asyncio.wait(
    #                        taskList, return_when=asyncio.FIRST_EXCEPTION
    #                    )
    #
    #                    await util.gracefulShutdown(taskList)
    #                    await asyncio.sleep(5)
    #
    #
    #            except asyncio.CancelledError:
    #                raise
    #            except Exception as e:
    #                self.logger.exception(f"Websocket Client 发生异常: {str(e)}")
    #                await asyncio.sleep(5)

    sensorReportGql = gql(
        """
        mutation reportUpdate($lightDataInput : LightDataInput!) {
            lightSelf {
                reportUpdate (lightDataInput: $lightDataInput) {
                    resultType
                }
            }
        }
        """
    )

    async def sensorReportLoop(self, session: AsyncClientSession):
        while True:
            try:
                data = await self.dataUpdateQueue.get()
                modbus = data["Modbus"]
                weather = modbus["Weather"]
                windSpeed = modbus["Wind_Speed"]

                await session.execute(
                    self.sensorReportGql,
                    {
                        "lightDataInput": {
                            "humidity": weather["Humidity"],
                            "temperature": weather["Temperature"],
                            "pm10": weather["PM10"],
                            "pm2_5": weather["PM2.5"],
                            "illumination": weather["Illuminance"],
                            "windSpeed": windSpeed["Wind_Speed"],
                            "windDirection": windSpeed["Wind_Direction"],
                        }
                    },
                )
            # except (asyncio.CancelledError, TransportError, WebSocketException):
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(f"sensorReportLoop exception: {str(e)}")
                await asyncio.sleep(5)

    stateReportGql = gql(
        """
        mutation reportState($lightState : LightStateInput){
          lightSelf {
            reportState(lightState : $lightState) {
              resultType
            }
          }
        }
        """
    )

    async def stateReportLoop(self, session: AsyncClientSession):
        while True:
            try:
                state = await self.stateUpdateQueue.get()

                await session.execute(
                    self.stateReportGql,
                    {
                        "$lightState": {
                            "enableWirelessCharging": state.enableWirelessCharging,
                            "wirelessChargingPower": state.wirelessChargingPower,
                        }
                    },
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(f"stateReportLoop exception: {str(e)}")
                await asyncio.sleep(5)

    configurationDistributionGql = gql(
        """
        subscription updateConfigurationEvent {
          updateConfigurationEvent  {
            key
            value
          }
        }
        """
    )

    async def configurationDistributionLoop(self, session: AsyncClientSession):
        while True:
            try:
                async for result in session.subscribe(
                        self.configurationDistributionGql
                ):
                    message = result["updateConfigurationEvent"]
                    key = message["key"]
                    value = message["value"]
                    await self.main.configureComponent.setConfigure(key, value)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(
                    f"configurationDistributionLoop exception: {str(e)}"
                )
                await asyncio.sleep(5)

    loginGql = """
        mutation login($username: String!, $password: String!) {
            login(username: $username, password: $password)
        }
        """

    detectionReportGql = """
        mutation ($detectionInput : DetectionInput!, $lightName : String!){
          self {
            getLightByName(name : $lightName) {
              reportDetection(detectionInput : $detectionInput) {
              	resultType
              }
            }
          }
        }
        """

    async def detectionReportLoop(self):

        while True:
            try:

                async with aiohttp.ClientSession() as session:

                    data = aiohttp.FormData()

                    data.add_field(
                        "operations",
                        json.dumps(
                            {
                                "query": self.loginGql,
                                "variables": {
                                    "username": self.username,
                                    "password": self.password,
                                },
                            }
                        ),
                        content_type="application/json",
                    )

                    async with session.post(
                            self.httpUrl,
                            data=data,
                    ) as response:

                        if response.status != 200:
                            raise Exception(f"http error: {str(response.status)}")

                        result = await response.json()
                        if "errors" in result and result["errors"] is not None:
                            raise Exception(f"result errors: \n{result}")

                        if "data" not in result or "login" not in result["data"]:
                            raise Exception(f"result data not found: \n{result}")

                        jwt = result["data"]["login"]
                        if jwt is None:
                            raise Exception(f"not obtained jwt")

                    res: detection.Result = await self.identifyKeyframeQueue.get()

                    if len(res.cellMap) == 0:
                        continue

                    height, width = res.inputImage.shape[:2]

                    detections = []

                    model: detection.Model
                    cells: list[detection.Cell]
                    cell: detection.Cell

                    for model, cells in res.cellMap.items():

                        modelName: str = model.name

                        for cell in cells:
                            box: Box = cell.box.normalization(width, height)
                            detections.append(
                                {
                                    "x": float(box.x),
                                    "y": float(box.y),
                                    "w": float(box.w),
                                    "h": float(box.h),
                                    "probability": float(cell.probability),
                                    "model": modelName,
                                    "item": cell.item.name,
                                }
                            )

                    if len(detections) == 0:
                        continue

                    data = aiohttp.FormData()

                    data.add_field(
                        "operations",
                        json.dumps(
                            {
                                "query": self.detectionReportGql,
                                "variables": {
                                    "detectionInput": {
                                        "items": detections,
                                        "image": None,
                                    },
                                    "lightName": self.localName,
                                },
                            }
                        ),
                        content_type="application/json",
                    )

                    data.add_field(
                        "map",
                        json.dumps({"image": ["variables.detectionInput.image"]}),
                        content_type="application/json",
                    )

                    image: cv2.typing.MatLike = res.inputImage

                    if len(image.shape) == 3:  # 彩色图像
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = await asyncio.get_event_loop().run_in_executor(
                            None, cv2.cvtColor, image, cv2.COLOR_BGR2RGB, None, 3
                        )

                    # _, jpeg_buffer = cv2.imencode(".jpg", img_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    _, jpeg_buffer = await asyncio.get_event_loop().run_in_executor(
                        None,
                        cv2.imencode,
                        ".jpg",
                        image,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 90],
                    )
                    jpeg_bytes = jpeg_buffer.tobytes()
                    image_file = BytesIO(jpeg_bytes)

                    data.add_field(
                        "image",
                        image_file,
                        filename="image.jpg",
                        content_type="image/jpeg",
                    )

                    async with session.post(
                            self.httpUrl, data=data, headers={"Authorization": f"{jwt}"}
                    ) as response:

                        if response.status != 200:
                            raise Exception(f"http error: {str(response.status)}")

                        result = await response.json()
                        if "errors" in result and result["errors"] is not None:
                            raise Exception(f"result errors: \n{result}")

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(f"上传关键帧时发生异常: {str(e)}")
                await asyncio.sleep(5)
