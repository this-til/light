#!/usr/bin/python3

import logging
import asyncio
import numpy as np
import cv2
import detection
import time
from asyncio.subprocess import PIPE
from util import Broadcaster, FFmpegPush

logger = logging.getLogger(__name__)

ip = "192.168.117.100"
rtsp_port = "554"
user = "admin"
password = "qWERTYUIOP"

width, height = 2560, 1440
fps = 25

cameraRtspUrl = f"rtsp://{user}:{password}@{ip}:{rtsp_port}/Streaming/Channels/101"

source: Broadcaster[cv2.typing.MatLike] = Broadcaster()
out: Broadcaster[cv2.typing.MatLike] = Broadcaster()
identifyKeyframe: Broadcaster[detection.Result] = Broadcaster()

cap: cv2.VideoCapture

pushRtspUrl = "rtsp://localhost:8554/stream1"


async def releaseCap():
    global cap
    if cap and cap.isOpened():
        await asyncio.get_event_loop().run_in_executor(None, cap.release)
    cap = None
    pass


async def readFrames():
    global cap

    while True:

        try:

            logger.info("尝试连接摄像头...")

            # gst_pipeline = f"rtspsrc location={pushRtspUrl} latency=0 ! rtph264depay ! h264parse ! nvh264dec ! videoconvert ! appsink"
            # cap = await asyncio.get_event_loop().run_in_executor(None, cv2.VideoCapture, gst_pipeline, cv2.CAP_GSTREAMER)

            cap = await asyncio.get_event_loop().run_in_executor(
                None, cv2.VideoCapture, cameraRtspUrl
            )

            if not cap.isOpened():
                logger.warning("连接失败，5秒后重试...")
                await asyncio.sleep(5)
                continue

            while True:
                ret, frame = await asyncio.get_event_loop().run_in_executor(
                    None, cap.read
                )

                if not ret:
                    logger.error("读取帧失败，释放资源并重连...")
                    raise Exception("读取帧失败...")

                h, w = frame.shape[:2]
                if w < h:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                await source.publish(frame)

        except asyncio.CancelledError:
            logger.info("任务被取消，执行清理...")
            await releaseCap()
            raise

        except Exception as e:
            logger.exception(f"发生未处理异常:  {str(e)}")
            await releaseCap()
            logger.info("5秒后尝试重新连接摄像头...")
            await asyncio.sleep(5)

        pass


async def renderFrames():
    while True:
        try:
            pass
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception(f"渲染帧时发生异常: {str(e)}")
        pass
    pass


async def pushFrames():
    await FFmpegPush(
        width,
        height,
        fps,
        pushRtspUrl,
        await out.subscribe(asyncio.Queue(maxsize=16)),
        __name__,
    ).pushFrames()
    pass


async def handleFrames():

    framesQueue: asyncio.Queue[cv2.typing.MatLike] = asyncio.Queue(maxsize=16)
    await source.subscribe(framesQueue)

    res: detection.Result | None = None
    task: asyncio.Task | None = None

    while True:
        try:

            sourceFrame = await framesQueue.get()

            if res is None:
                res = detection.Result(sourceFrame, {})

            if task is None or task.done():

                if task is not None:
                    res = task.result()
                    await identifyKeyframe.publish(res)

                def _runDetection(inputImage, useModel):
                    start_time = time.perf_counter()
                    res = detection.runDetection(inputImage, useModel)
                    end_time = time.perf_counter()
                    duration_ms = (end_time - start_time) * 1000
                    logger.info(f"inference 耗时: {duration_ms:.3f}ms")
                    return res

                task = asyncio.get_event_loop().run_in_executor(
                    None,
                    _runDetection,
                    sourceFrame,
                    detection.models,
                )

            _res = detection.Result(sourceFrame, res.cellMap)
            await out.publish(await _res.drawOutputImageAsunc())

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception(f"处理帧时发生异常: {str(e)}")

            res = None
            if task is not None:
                task.cancel()
                task = None
        pass

    pass


async def initCamera():
    logger.info("初始化摄像头模块")
    asyncio.create_task(readFrames())
    asyncio.create_task(handleFrames())
    asyncio.create_task(pushFrames())


async def releaseCamera():
    logger.info("释放摄像头资源")
