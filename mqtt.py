import asyncio
import logging
import json
import paho.mqtt.client as mqtt
import main
from main import Component, ConfigField

logger = logging.getLogger(__name__)


class MqttComponent(Component):

    topic: ConfigField[str] = ConfigField()
    clientId: ConfigField[str] = ConfigField()
    ip: ConfigField[str] = ConfigField()
    port: ConfigField[int] = ConfigField()
    userName: ConfigField[str] = ConfigField()
    password: ConfigField[str] = ConfigField()

    client: mqtt.Client = None  # type: ignore

    async def init(self):
        await super().init()

        self.client = mqtt.Client(client_id=self.clientId)

        self.client.subscribe(self.topic)

        self.client.on_connect = self.onConnect
        self.client.on_message = self.onMessage

        self.client.username_pw_set(self.userName, self.password)
        self.client.connect(self.ip, self.port)
        self.client.loop_start()  # 使用非阻塞循环

        self.client.publish(self.topic, json.dumps({"温度": 23}))

    def onConnect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.debug("成功连接至MQTT服务器!")
        else:
            logger.debug(f"连接失败，错误码: {rc}")

    def onMessage(self, client, userdata, msg):
        logger.debug(f"收到消息: Topic={msg.topic}, Payload={msg.payload.decode()}")
