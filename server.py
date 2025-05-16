#!/usr/bin/python3

from quart import Quart, jsonify, request
import asyncio
import threading
from hypercorn.asyncio import serve
from hypercorn.config import Config
import logging
import device
import configure
from quart_cors import cors
from enum import Enum

logger = logging.getLogger(__name__)



app = Quart(__name__)
app = cors(app, allow_origin="*") 



# 自定义配置示例
class ServerConfig:
    host = "0.0.0.0"
    port = 8000
    workers = 2
    debug = True
    
async def runServer(config: ServerConfig = ServerConfig()):

    quart_config = Config()
    quart_config.bind = [f"{config.host}:{config.port}"]
    quart_config.workers = config.workers
    quart_config.debug = config.debug
    
    try:
        await serve(app, quart_config)
    except KeyboardInterrupt:
        logger.info("服务器关闭中...")
            
    pass
    
    
async def initServer():
    pass

async def releaseServer():
    await app.shutdown()
    pass


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
    return createCarrier(device.deviceValue)


@app.route("/getDeviceValue")
async def getDeviceValue():
    key : str = str(request.args.get("key"))
    return createCarrier(device.getDeviceValue(key))


@app.route("/setDeviceValue", methods=["POST"])
async def setDeviceValue():
    key : str = str(request.args.get("key"))
    data = await request.get_json()
    value = data.get("value")
    return createCarrier(device.setDeviceValue(key, value))


@app.route("/getConfigure")
async def getConfigure():
    key : str = str(request.args.get("key"))
    return createCarrier(configure.getConfigure(key))


@app.route("/setConfigure", methods=["POST"])
async def setConfigure():
    key : str = str(request.args.get("key"))
    data = await request.get_json()
    value = data.get("value")
    return createCarrier(await configure.setConfigure(key, value))


@app.errorhandler(Exception)
async def http_exception_handler(e):
    return createCarrier(None, status=Status.FAIL, message=e.description), e.code
