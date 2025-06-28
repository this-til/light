import asyncio
import time
from typing import Optional

import gpiod

from main import Component, ConfigField
import util

# 默认配置
BUTTON_GPIO_BANK = 3  # GPIO 组 (3)
BUTTON_GPIO_LETTER = 'B'  # 组内字母 (B)
BUTTON_GPIO_PIN = 3  # 引脚号 (3)

DEFAULT_DEBOUNCE_TIME = 0.05  # 防抖时间(秒)


class ButtonPressEvent:
    """按钮按下事件"""

    def __init__(self, timestamp: float, gpio_config: 'GpioConfig'):
        self.timestamp = timestamp
        self.gpio_config = gpio_config


class GpioConfig:
    """GPIO配置类"""

    def __init__(self, bank: int, letter: str, pin: int):
        self.bank = bank
        self.letter = letter.upper()
        self.pin = pin
        self._validate_config()

    def _validate_config(self):
        """验证GPIO配置"""
        if self.letter not in ['A', 'B', 'C', 'D']:
            raise ValueError(f"Invalid GPIO letter: {self.letter}")
        if not (0 <= self.pin <= 31):
            raise ValueError(f"Invalid GPIO pin: {self.pin}")
        if not (0 <= self.bank <= 15):
            raise ValueError(f"Invalid GPIO bank: {self.bank}")

    @property
    def calcOffset(self) -> int:
        """计算GPIO全局偏移量的getter方法"""
        letter_map = {
            'A': 0,
            'B': 1,
            'C': 2,
            'D': 3
        }
        letter_value = letter_map[self.letter]
        return (self.bank * 32) + (letter_value * 8) + self.pin

    @property
    def lineOffset(self) -> int:
        """计算相对于芯片的线偏移量"""
        return self.calcOffset - self.bank * 32

    def __str__(self) -> str:
        return f"GPIO(Bank:{self.bank}, Letter:{self.letter}, Pin:{self.pin})"


def asGpioConfig(map: dict[str, str]) -> GpioConfig:
    return GpioConfig(int(map['bank']), map['letter'], int(map['pin']))


class GpioComponent(Component):
    buttonGpioConfigMap: ConfigField[dict[str, str]] = ConfigField()
    colorOutGpioConfigMap: ConfigField[dict[str, str]] = ConfigField()

    buttonGpioConfig: GpioConfig = None
    colorOutGpioConfig: GpioConfig = None

    buttonPress: util.Broadcaster[ButtonPressEvent] = util.Broadcaster()

    async def init(self):
        await super().init()
        buttonGpioConfig = asGpioConfig(self.buttonGpioConfigMap)
        colorOutGpioConfig = asGpioConfig(self.colorOutGpioConfigMap)

        asyncio.create_task(self.buttonMonitorLoop())

    async def buttonMonitorLoop(self):
        """按钮监听主循环"""
        pass

