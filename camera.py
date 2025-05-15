#!/usr/bin/python3

import logging
import asyncio
import numpy as np
import cv2
import detection
import time
import hkws
import ctypes
import sys
import util
import hkws_sdk
from ctypes import *
from typing import *
from asyncio.subprocess import PIPE
from util import Broadcaster, FFmpegPushFrame, ByteFFmpegPull
from enum import IntEnum, unique

logger = logging.getLogger(__name__)

ip = "192.168.117.100"
rtsp_port = "554"
port = 8000
user = "admin"
password = "qWERTYUIOP"
SDKPath = "/home/elf/HCNetSDKV6.1.9.45_build20220902_ArmLinux64_ZH/MakeAll/"

width, height = 2560, 1440
fps = 25

# 音频参数
FREQUENCY = 16000  # 采样率16kHz
SAMPLE_SIZE = -16  # 16位有符号PCM
CHANNELS = 1  # 单声道
BUFFER_SIZE = 4096  # 每次读取的数据块大小，须为2的倍数


cameraRtspUrl = f"rtsp://{user}:{password}@{ip}:{rtsp_port}/Streaming/Channels/101"

source: Broadcaster[cv2.typing.MatLike] = Broadcaster()
audioSource: Broadcaster[bytes] = Broadcaster()
out: Broadcaster[cv2.typing.MatLike] = Broadcaster()
identifyKeyframe: Broadcaster[detection.Result] = Broadcaster()


pushRtspUrl = "rtsp://localhost:8554/stream1"


cap: cv2.VideoCapture = None  # type: ignore

userId: int = -1
realHandle: int = -1
voiceHandle: int = -1


async def releaseCap():
    global cap
    if cap and cap.isOpened():
        await asyncio.get_event_loop().run_in_executor(None, cap.release)
    cap = None  # type: ignore
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
                ret: bool
                frame: cv2.typing.MatLike

                ret, frame = await asyncio.get_event_loop().run_in_executor(
                    None, cap.read
                )

                if not ret:
                    logger.error("读取帧失败，释放资源并重连...")
                    raise Exception("读取帧失败...")

                h, w = frame.shape[:2]
                if w < h:
                    frame = await asyncio.get_event_loop().run_in_executor(
                        None, cv2.rotate, frame, cv2.ROTATE_90_CLOCKWISE
                    )

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


async def extractAudio():

    ffmpeg_command = [
        "ffmpeg",
        "-i",
        cameraRtspUrl,  # 输入RTSP流
        "-vn",  # 忽略视频
        "-acodec",
        "pcm_s16le",  # 输出PCM格式
        "-ar",
        str(FREQUENCY),  # 采样率
        "-ac",
        str(CHANNELS),  # 声道数
        "-f",
        "s16le",  # 输出格式为s16le
        "-loglevel",
        "quiet",  # 屏蔽FFmpeg日志
        "pipe:1",  # 输出到标准输出
    ]

    async def publishToAudio(b: bytes):
        await audioSource.publish(b)

    await ByteFFmpegPull(
        ffmpeg_command, BUFFER_SIZE, publishToAudio, "cameraAudio"
    ).loop()

    pass


async def pushFrames():
    await FFmpegPushFrame(
        width,
        height,
        fps,
        pushRtspUrl,
        await out.subscribe(asyncio.Queue(maxsize=16)),
        __name__,
    ).loop()
    pass


async def handleFrames():

    framesQueue: asyncio.Queue[cv2.typing.MatLike] = asyncio.Queue(maxsize=16)
    await source.subscribe(framesQueue)

    res: detection.Result | None = None
    task: asyncio.Future[detection.Result] | None = None

    while True:
        try:

            sourceFrame: cv2.typing.MatLike = await framesQueue.get()

            if res is None:
                res = detection.Result(sourceFrame, {})

            if task is None or task.done():

                if task is not None:
                    res = task.result()
                    await identifyKeyframe.publish(res)

                def _runDetection(
                    inputImage: cv2.typing.MatLike, useModel: list[detection.Model]
                ):
                    # start_time = time.perf_counter()
                    res = detection.runDetection(inputImage, useModel)
                    # end_time = time.perf_counter()
                    # duration_ms = (end_time - start_time) * 1000
                    # logger.info(f"inference 耗时: {duration_ms:.3f}ms")
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

    global  userId, realHandle, voiceHandle

    hkws_sdk.initSdk()
    hkws_sdk.setConnectTime()
    hkws_sdk.setReconnect()
    userId = hkws_sdk.login(ip, port, user, password)
    realHandle = hkws_sdk.realPlay(userId)
    voiceHandle = hkws_sdk.startVoiceCom(userId)
    # camera.ptzControlOther(1, DeviceCommand.TILT_DOWN, 0)

    # asyncio.create_task(readFrames())
    # asyncio.create_task(extractAudio())
    asyncio.create_task(handleFrames())
    asyncio.create_task(pushFrames())


async def releaseCamera():
    logger.info("释放摄像头资源")
    hkws_sdk.stopPreview(realHandle)
    hkws_sdk.stopVoiceCom(voiceHandle)
    hkws_sdk.logout(userId)
    hkws_sdk.sdkClean()
