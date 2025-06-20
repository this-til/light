#!/usr/bin/python3

from __future__ import annotations

import os
import asyncio
import json
from pathlib import Path

import util
from main import Component

CONFIG_FILE_PATH = "config.json"


class ConfigureChangeEvent:

    key: str = None  # type: ignore
    value: object | None = None
    oldValue: object | None = None

    def __init__(self, key: str, value: object | None, oldValue: object | None):
        self.key = key
        self.value = value
        self.oldValue = oldValue


class ConfigureComponent(Component):

    needSave: asyncio.Event = asyncio.Event()
    configureMap = {}

    configureChange: util.Broadcaster[ConfigureChangeEvent] = util.Broadcaster()

    def __init__(self):
        super().__init__()
        self.needSave.clear()

    async def awakeInit(self):
        await super().awakeInit()

        try:
            # 尝试加载现有配置文件
            with open(f"{os.path.dirname(os.path.abspath(__file__))}/{CONFIG_FILE_PATH}", "r") as file:
                self.configureMap = json.load(file)
                self.logger.info(f"Loaded config from {CONFIG_FILE_PATH}")

        except FileNotFoundError:
            # 生成默认配置
            self.logger.warning(
                f"Config file not found, creating default at {CONFIG_FILE_PATH}"
            )

            # 确保目录存在
            config_path = Path(CONFIG_FILE_PATH)
            config_path.parent.mkdir(parents=True, exist_ok=True)

            # 写入默认配置
            with open(CONFIG_FILE_PATH, "w") as file:
                json.dump({}, file, indent=2)
                self.logger.info(f"Created default config at {CONFIG_FILE_PATH}")

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON format: {e}")

        except Exception as e:
            self.logger.critical(f"Unexpected config error: {e}")

        pass

    async def initEnd(self):
        await super().initEnd()

        defConfig = util.flattenJson(self.configureMap)
        for k, v in defConfig.items():
            await self.configureChange.publish(ConfigureChangeEvent(k, v, None))
            await asyncio.sleep(0.1)

        asyncio.create_task(self.saveConfigureLoop())

    def getPriority(self) -> int:
        return 1 << 12

    def getConfigure(self, key: str):
        return util.getFromJson(key, self.configureMap)

    async def setConfigure(self, key: str, value):
        old = util.getFromJson(key, self.configureMap)
        util.setFromJson(key, value, self.configureMap)
        await self.configureChange.publish(ConfigureChangeEvent(key, value, old))
        self.needSave.set()

    async def saveConfigure(self):
        with open(CONFIG_FILE_PATH, "w") as file:
            json.dump(self.configureMap, file, indent=4)
            self.logger.info(
                f"Configuration saved to {CONFIG_FILE_PATH}: {self.configureMap}"
            )
            self.needSave.set()

    async def saveConfigureLoop(self):

        while True:
            try:
                await self.needSave.wait()
                await self.saveConfigure()
                self.needSave.clear()
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(f"保存配置文件时发生异常:{e} ")
                await asyncio.sleep(5)
