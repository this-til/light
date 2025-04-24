#!/usr/bin/python3

import asyncio
import logging

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
