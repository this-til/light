#!/usr/bin/python3

import asyncio
import logging
import json
import util
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_FILE_PATH = "/home/elf/light/config.json"

needSave = False

configureMap = {}


class ConfigureChangeEvent:

    key: str = None  # type: ignore
    value: object | None = None
    oldValue: object | None = None

    def __init__(self, key: str, value: object | None, oldValue: object | None):
        self.key = key
        self.value = value
        self.oldValue = oldValue

    pass


configureChange: util.Broadcaster[ConfigureChangeEvent] = util.Broadcaster()


async def saveConfigure():
    global configureMap, needSave
    with open(CONFIG_FILE_PATH, "w") as file:
        json.dump(configureMap, file, indent=4)
        logger.info(f"Configuration saved to {CONFIG_FILE_PATH}: {configureMap}")
        needSave = False


async def saveConfigureLoop():
    global needSave

    while True:
        try:
            await asyncio.sleep(1)
            if needSave:
                await saveConfigure()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception(f"保存配置文件时发生异常:{e} ")
            await asyncio.sleep(5)


async def initConfigure():
    global configureMap

    try:
        # 尝试加载现有配置文件
        with open(CONFIG_FILE_PATH, "r") as file:
            configureMap = json.load(file)
            logger.info(f"Loaded config from {CONFIG_FILE_PATH}")

    except FileNotFoundError:
        # 生成默认配置
        logger.warning(f"Config file not found, creating default at {CONFIG_FILE_PATH}")

        # 确保目录存在
        config_path = Path(CONFIG_FILE_PATH)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # 写入默认配置
        with open(CONFIG_FILE_PATH, "w") as file:
            json.dump({}, file, indent=2)
            logger.info(f"Created default config at {CONFIG_FILE_PATH}")

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {e}")
        raise RuntimeError("Configuration file format error") from e

    except Exception as e:
        logger.critical(f"Unexpected config error: {e}")
        raise RuntimeError("Failed to initialize configuration") from e

    asyncio.create_task(saveConfigureLoop())

    return configureMap


async def releaseConfigure():
    if needSave:
        await saveConfigure()
    pass


def getConfigure(key: str):
    return util.getFromJson(key, configureMap)


async def setConfigure(key: str, value):
    global needSave
    old = util.getFromJson(key, configureMap)
    util.setFromJson(key, value, configureMap)
    await configureChange.publish(ConfigureChangeEvent(key, value, old))
    needSave = True
