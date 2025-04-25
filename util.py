#!/usr/bin/python3

import asyncio
import logging
import json
from typing import Generic, TypeVar, Optional

logger = logging.getLogger(__name__)


def getAllTasks() -> list[asyncio.Task]:
    """
    获取当前事件循环中的所有异步任务，并且排除当前调用的
    """
    loop = asyncio.get_event_loop()
    tasks = asyncio.all_tasks(loop)
    current_task = asyncio.current_task(loop)
    return [t for t in tasks if t is not current_task]


async def gracefulShutdown():
    """
    优雅关闭所有异步任务
    """
    tasks = getAllTasks()
    for task in tasks:
        task.cancel()  # 发送取消信号

    for task in tasks:
        try:
            await task  # 等待任务完成
        except asyncio.CancelledError:
            logging.error(f"Task {task.get_name()} was cancelled.")
        except Exception as e:
            logging.error(f"Task {task.get_name()} raised an exception:", e)


T = TypeVar("T")  # 定义泛型类型


class Broadcaster(Generic[T]):  # 继承 Generic 标记泛型类型
    def __init__(self):
        self.queues: list[asyncio.Queue[T]] = []  # 明确队列存储类型
        self.lock = asyncio.Lock()

    async def subscribe(
        self, queue: asyncio.Queue[T]
    ) -> None:  # 订阅的队列类型与泛型一致
        async with self.lock:
            self.queues.append(queue)

    async def unsubscribe(self, queue: asyncio.Queue[T]) -> None:
        async with self.lock:
            self.queues.remove(queue)

    async def publish(self, item: T) -> None:  # 发布项类型与泛型一致
        async with self.lock:
            for q in self.queues:
                await q.put(item)


def flattenJson(data, parent_key="", sep=".", list_sep="[{}]"):
    items = {}
    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            items.update(flattenJson(value, new_key, sep, list_sep))
    elif isinstance(data, (list, tuple)):
        for index, item in enumerate(data):
            new_key = f"{parent_key}{list_sep.format(index)}"
            items.update(flattenJson(item, new_key, sep, list_sep))
    else:
        if parent_key:  # 忽略根元素为非容器的值
            items[parent_key] = data
    return items


def jsonDeepMerge(source, overrides):
    '''
    JSON 深度合并（直接修改 source 字典）
    警告：此函数会直接修改传入的 source 参数！
    '''
    for key, value in overrides.items():
        # 若双方均为字典，则递归合并
        if isinstance(value, dict) and isinstance(source.get(key), dict):
            jsonDeepMerge(source[key], value)
        # 否则直接覆盖（或添加）键值
        else:
            source[key] = value
    return source  # 返回修改后的 source