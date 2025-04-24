#!/usr/bin/python3

import logging
import asyncio
import numpy as np
import cv2
from asyncio.subprocess import PIPE

import util
import main

logger = logging.getLogger(__name__)

ip = "192.168.117.100"
rtsp_port = "554"
user = "admin"
password = "qWERTYUIOP"

width, height = 2560, 1140
fps = 25

cameraRtspUrl = f"rtsp://{user}:{password}@{ip}:{rtsp_port}/Streaming/Channels/101"

source: asyncio.Queue[cv2.typing.MatLike] = asyncio.Queue(maxsize=16)
out: asyncio.Queue[cv2.typing.MatLike] = asyncio.Queue(maxsize=16)

cap: cv2.VideoCapture = None

pushRtspUrl = "rtsp://localhost:8554/channels001"

pushProcess: asyncio.subprocess.Process = None


async def read_frames():
    global cap
    loop = asyncio.get_event_loop()

    while True:
        try:
            # 连接/重连逻辑
            if cap is None or not cap.isOpened():
                logger.info("尝试连接摄像头...")
                # 在executor中同步执行VideoCapture初始化
                new_cap = await loop.run_in_executor(
                    None, cv2.VideoCapture, cameraRtspUrl
                )
                if not new_cap.isOpened():
                    logger.warning("连接失败，5秒后重试...")
                    await asyncio.sleep(5)
                    continue

                # 成功连接后替换旧cap
                if cap is not None:
                    logger.debug("释放旧摄像头资源")
                    await loop.run_in_executor(None, cap.release)
                cap = new_cap
                logger.info("摄像头连接成功")

            # 在executor中同步读取帧
            ret, frame = await loop.run_in_executor(None, cap.read)

            if not ret:
                logger.error("读取帧失败，释放资源并重连...")
                await loop.run_in_executor(None, cap.release)
                cap = None
                continue

            # 队列处理（非阻塞替换旧帧）
            if source.full():
                try:
                    source.get_nowait()
                    logger.debug("队列已满，丢弃最旧帧")
                except asyncio.QueueEmpty:
                    pass
            await source.put(frame)

        except asyncio.CancelledError:
            logger.info("任务被取消，执行清理...")
            if cap and cap.isOpened():
                await loop.run_in_executor(None, cap.release)
            cap = None
            raise
        except Exception as e:
            logger.exception(f"发生未处理异常:  {str(e)}")
            if cap and cap.isOpened():
                await loop.run_in_executor(None, cap.release)
            cap = None
            logger.info("5秒后尝试重新连接摄像头...")
            await asyncio.sleep(5)


async def push_frames():

    command = [
        "ffmpeg",
        "-y",  # 覆盖输出文件
        "-f",
        "rawvideo",  # 输入格式为原始视频流
        "-pix_fmt",
        "bgr24",  # OpenCV 的 BGR 格式
        "-s",
        f"{width}x{height}",  # 分辨率
        "-r",
        f"{fps}",  # 帧率
        "-i",
        "-",  # 标准输入
        "-c:v",
        "libx264",  # 视频编码器（H.264）
        "-preset",
        "ultrafast",  # 编码速度（更快的实时编码）
        "-pix_fmt",
        "yuv420p",  # 输出像素格式（兼容性更好）
        "-f",
        "rtsp",  # 输出协议为 RTSP
        "-rtsp_transport",
        "tcp",  # 使用 TCP 协议传输
        pushRtspUrl,  # 目标地址
    ]

    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )

    while True:
        frame = await out.get()
        data = frame.tobytes()
        process.stdin.write(data)
        await process.stdin.drain()
        pass


async def initCamera():
    logger.info("初始化摄像头模块")
    asyncio.create_task(read_frames())
    asyncio.create_task(push_frames())


async def releaseCamera():
    logger.info("释放摄像头资源")
    if cap and cap.isOpened():
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, cap.release)
    if pushProcess:
        try:
            pushProcess.stdin.close()
            await pushProcess.wait()
        except Exception as e:
            logger.exception(f"关闭推流进程失败: {str(e)}")
