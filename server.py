from quart import Quart, jsonify
import asyncio
from hypercorn.asyncio import serve
from hypercorn.config import Config
import logging
import device
from enum import Enum

logger = logging.getLogger(__name__)
app = Quart(__name__)


# 自定义配置示例
class ServerConfig:
    host = "0.0.0.0"
    port = 8000
    workers = 2
    debug = True


async def initServer(config: ServerConfig = ServerConfig()):
    quart_config = Config()
    quart_config.bind = [f"{config.host}:{config.port}"]
    quart_config.workers = config.workers
    quart_config.debug = config.debug

    # 启动服务器任务
    asyncio.create_task(serve(app, quart_config))
    logger.info(f"🚀 服务器已启动在 {config.host}:{config.port}")


async def releaseServer():
    pass


class Status(Enum):
    SUCCESS = "SUCCESS"
    FAIL = "FAIL"
    ERROR = "ERROR"


def createCarrier(data, status: Enum = Status.SUCCESS, message: str = None):
    if message is None:
        message = status.value
    return jsonify({"message": message, "data": data, "status": status.value})


# 示例路由
@app.route("/getDeviceValue")
async def get_status():
    return createCarrier(device.deviceValue)