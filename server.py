#!/usr/bin/python3

from quart import Quart, jsonify, request
import asyncio
import threading
from hypercorn.asyncio import serve
from hypercorn.config import Config
import logging
import configure
from quart_cors import cors
from enum import Enum
from main import Component, ConfigField

logger = logging.getLogger(__name__)

app = Quart(__name__)
app = cors(app, allow_origin="*")

serverComponent: "ServerComponent" = None  # type: ignore


class Status(Enum):
    SUCCESS = "SUCCESS"
    FAIL = "FAIL"
    ERROR = "ERROR"


def createCarrier(data, status: Enum = Status.SUCCESS, message: str | None = None):
    if message is None:
        message = status.value
    return jsonify({"message": message, "data": data, "status": status.value})


@app.route("/getDeviceValues")
async def getDeviceValues():
    return createCarrier(serverComponent.main.deviceComponent.deviceValue)


@app.route("/getDeviceValue")
async def getDeviceValue():
    key: str = str(request.args.get("key"))
    return createCarrier(serverComponent.main.deviceComponent.getDeviceValue(key))


@app.route("/setDeviceValue", methods=["POST"])
async def setDeviceValue():
    key: str = str(request.args.get("key"))
    data = await request.get_json()
    value = data.get("value")
    return createCarrier(serverComponent.main.detectionComponent.setDeviceValue(key, value))


@app.route("/getConfigure")
async def getConfigure():
    key: str = str(request.args.get("key"))
    return createCarrier(serverComponent.main.configureComponent.getConfigure(key))


@app.route("/setConfigure", methods=["POST"])
async def setConfigure():
    key: str = str(request.args.get("key"))
    data = await request.get_json()
    value = data.get("value")
    return createCarrier(
        await serverComponent.main.configureComponent.setConfigure(key, value)
    )


@app.errorhandler(Exception)
async def http_exception_handler(e):
    return createCarrier(None, status=Status.FAIL, message=e.description), e.code


class ServerComponent(Component):
    host: ConfigField[str] = ConfigField()
    port: ConfigField[int] = ConfigField()
    workers: ConfigField[int] = ConfigField()
    debug: ConfigField[bool] = ConfigField()

    def __init__(self):
        super().__init__()
        global serverComponent
        serverComponent = self

    async def runServer(self):
        quart_config = Config()
        quart_config.bind = [f"{self.host}:{self.port}"]
        quart_config.workers = self.workers
        quart_config.debug = self.debug

        try:
            # await serve(app, quart_config)
            await asyncio.create_task(serve(app, quart_config))
        except KeyboardInterrupt:
            logger.info("服务器关闭中...")

        pass
