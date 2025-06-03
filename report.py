import asyncio
import logging
import json
import paho.mqtt.client as mqtt
import detection
import main
from typing import cast
from gql import gql, Client
from gql.transport.websockets import WebsocketsTransport
from websockets import Subprotocol

from main import Component, ConfigField

logger = logging.getLogger(__name__)


class MqttReportComponent(Component):
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

        self.client = mqtt.Client(client_id=self.clientId)

        self.client.subscribe(self.receiveTopic)
        self.client.subscribe(self.updataTopic)

        self.client.on_connect = self.onConnect
        self.client.on_message = self.onMessage

        self.client.username_pw_set(self.userName, self.password)
        self.client.connect(self.ip, self.port)
        self.client.loop_start()  # 使用非阻塞循环

        asyncio.create_task(self.mqttReportLoop())

    async def mqttReportLoop(self):

        event = await self.main.deviceComponent.dataUpdate.subscribe(asyncio.Event())

        while True:

            try:
                await event.wait()
                event.clear()
                data = self.main.deviceComponent.deviceValue
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
    username: ConfigField[str] = ConfigField()
    password: ConfigField[str] = ConfigField()
    localName: ConfigField[str] = ConfigField()

    url: ConfigField[str] = ConfigField()
    headers: ConfigField[dict] = ConfigField()
    timeout: ConfigField[int] = ConfigField()
    subprotocols: ConfigField[str] = ConfigField()

    async def init(self):
        asyncio.create_task(self.sensorReportLoop())
        pass

    def establishLink(self) -> WebsocketsTransport:

        return WebsocketsTransport(
            url=self.url,
            headers=self.headers,
            init_payload={
                "username": self.username,
                "password": self.password,
                "linkType": "WEBSOCKET_LIGHT",
                "deviceName": self.localName,
            },
            keep_alive_timeout=self.timeout,
            subprotocols=[cast(Subprotocol, self.subprotocols)],
        )

    sensorReportGql = gql(
        """
        query report(
            $humidity : Float,
            $temperature : Float,
            $pm10 : Float,
            $pm2_5 : Float,
            $illumination : Float,
            $windSpeed : Float,
            $windDirection : Float
        )
        {
            lightSelf {
                name
                reportUpdata (
                    lightDataInput: {
                        humidity : $humidity
                        temperature : $temperature
                        pm10 : $pm10
                        pm2_5 : $pm2_5
                        illumination : $illumination
                        windSpeed : $windSpeed
                        windDirection : $windDirection
                    }
                ) {
                    resultType
                }
            }
        }
    """
    )

    async def sensorReportLoop(self):
        event = await self.main.deviceComponent.dataUpdate.subscribe(asyncio.Event())

        while True:

            try:

                ws = self.establishLink()

                async with Client(
                        transport=ws,
                        fetch_schema_from_transport=False
                ) as session:

                    await event.wait()
                    event.clear()

                    data = self.main.deviceComponent.deviceValue
                    modbus = data["Modbus"]
                    weather = modbus["Weather"]
                    windSpeed = modbus["Wind_Speed"]

                    await session.execute(
                        self.sensorReportGql,
                        {
                            "humidity": weather["Humidity"],
                            "temperature": weather["Temperature"],
                            "pm10": weather["PM10"],
                            "pm2_5": weather["PM2.5"],
                            "illumination": weather["Illuminance"],
                            "windSpeed": windSpeed["Wind_Speed"],
                            "windDirection": windSpeed["Wind_Direction"]
                        }
                    )

                    await asyncio.sleep(5)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(f"上传数据时发生异常: {str(e)}")
                await asyncio.sleep(5)

    async def detectionReportLoop(self):

        queue: asyncio.Queue[detection.Result] = await self.main.cameraComponent.identifyKeyframe.subscribe(
            asyncio.Queue(maxsize=1)
        )

        while True:
            try:

                res = await queue.get()



            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(f"上传关键帧时发生异常: {str(e)}")

    async def configurationDistributionLoop(self):



        pass
