import asyncio
import logging
import json
import paho.mqtt.client as mqtt
import main
from main import Component, ConfigField


logger = logging.getLogger(__name__)


class MqttComponent(Component):

    receiveTopic: ConfigField[str] = ConfigField()
    updataTopic : ConfigField[str] = ConfigField()
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

        asyncio.create_task(self.upDateLoop())

    async def upDateLoop(self):

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
                            "光照" : weather["Illuminance"],
                        
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
