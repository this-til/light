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

    stateUpdateQueue: asyncio.Queue[dict] = asyncio.Queue(maxsize=8)
    detectionKeyframeQueue: asyncio.Queue[detection.Result] = asyncio.Queue(maxsize=8)
    sustainedDetectionKeyframeQueue: asyncio.Queue[detection.Result] = asyncio.Queue(maxsize=8)

    async def init(self):
        if self.enable:
            await self.main.stateComponent.stateChange.subscribe(self.stateUpdateQueue)
            await self.main.orbbecCameraComponent.detectionKeyframe.subscribe(
                self.detectionKeyframeQueue
            )
            await  self.main.orbbecCameraComponent.sustainedDetectionKeyframe.subscribe(
                self.sustainedDetectionKeyframeQueue
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
                "deviceType": "CAR",
                "deviceName": self.localName,
            },
            keep_alive_timeout=self.timeout,
            ping_interval=self.pingInterval,
            subprotocols=[cast(Subprotocol, self.subprotocol)],
        )

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
                asyncio.create_task(self.stateReportLoop(session)),
                asyncio.create_task(self.configurationDistributionLoop(session)),
                asyncio.create_task(self.sustainedDetectionReportLoop(session)),
                asyncio.create_task(self.commandDownEventLoop(session)),
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

    stateReportGql = gql(
        """
        mutation reportState($carState : CarStateInput){
          deviceSelf {
            asCar {
              reportState(carState : $carState) {
                resultType
              }
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
                        "carState": state
                        # "carState": {
                        #    "TODO": state.TODO,  # 根据schema，CarStateInput目前只有TODO字段
                        # }
                    },
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if not self.main.run:
                    raise
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
                if not self.main.run:
                    raise
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
            getDeviceByName(name : $lightName, deviceType: LIGHT) {
              reportDetection(detectionInput : $detectionInput) {
                resultType
              }
            }
          }
        }
        """

    def convertDetectionResultToItems(self, res: detection.Result) -> list[dict]:
        """Convert detection result to reportable items"""
        height, width = res.inputImage.shape[:2]
        items = []

        model: detection.Model
        cells: list[detection.Cell]
        cell: detection.Cell

        for model, cells in res.cellMap.items():
            modelName: str = model.name
            for cell in cells:
                box: Box = cell.box.normalization(width, height)
                items.append(
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
        return items

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

                    res: detection.Result = await self.detectionKeyframeQueue.get()
                    items = self.convertDetectionResultToItems(res)
                    if len(items) == 0:
                        continue

                    data = aiohttp.FormData()

                    data.add_field(
                        "operations",
                        json.dumps(
                            {
                                "query": self.detectionReportGql,
                                "variables": {
                                    "detectionInput": {
                                        "items": items,
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
                if not self.main.run:
                    raise
                self.logger.exception(f"上传关键帧时发生异常: {str(e)}")
                await asyncio.sleep(5)

    sustainedDetectionReportGql = gql(
        """
        mutation sustainedReportDetection($items: [DetectionItemInput!]!) {
            deviceSelf {
                sustainedReportDetection(items: $items) {
                    resultType
                }
            }
        }
        """
    )

    async def sustainedDetectionReportLoop(self, session: AsyncClientSession):
        while True:
            try:
                res: detection.Result = await self.sustainedDetectionKeyframeQueue.get()
                items = self.convertDetectionResultToItems(res)

                await session.execute(
                    self.sustainedDetectionReportGql,
                    {"items": items}
                )

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(f"sustainedDetectionReportLoop exception: {str(e)}")
                await asyncio.sleep(5)

    commandDownEventGql = gql(
        """
        subscription commandDownEvent {
            commandDownEvent {
                key
                value
            }
        }
        """
    )

    async def commandDownEventLoop(self, session: AsyncClientSession):
        """Handle incoming command down events"""
        while True:
            try:
                async for result in session.subscribe(self.commandDownEventGql):
                    message = result["commandDownEvent"]
                    key = message["key"]
                    value = message["value"]
                    await self.main.commandComponent.onCommand(key, value)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                if not self.main.run:
                    raise
                self.logger.exception(f"commandDownEventLoop exception: {str(e)}")
                await asyncio.sleep(5)
