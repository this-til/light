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

import main
from main import Component, ConfigField


logger = logging.getLogger(__name__)


# 音频参数
FREQUENCY = 16000  # 采样率16kHz
SAMPLE_SIZE = -16  # 16位有符号PCM
CHANNELS = 1  # 单声道
BUFFER_SIZE = 4096  # 每次读取的数据块大小，须为2的倍数


class CameraComponent(Component):

    ip: ConfigField[str] = ConfigField()
    rtspPort: ConfigField[int] = ConfigField()
    username: ConfigField[str] = ConfigField()
    password: ConfigField[str] = ConfigField()

    width: ConfigField[int] = ConfigField()
    height: ConfigField[int] = ConfigField()
    fps: ConfigField[int] = ConfigField()

    cameraRtspUrl: ConfigField[str] = ConfigField()

    pushRtspUrl: ConfigField[str] = ConfigField()

    sdkPort: ConfigField[int] = ConfigField()

    source: Broadcaster[cv2.typing.MatLike] = Broadcaster()
    audioSource: Broadcaster[bytes] = Broadcaster()
    out: Broadcaster[cv2.typing.MatLike] = Broadcaster()
    identifyKeyframe: Broadcaster[detection.Result] = Broadcaster()

    cap: cv2.VideoCapture = None  # type: ignore

    userId: int = -1
    realHandle: int = -1
    voiceHandle: int = -1

    async def init(self):
        self.userId = hkws_sdk.login(
            self.ip, self.sdkPort, self.username, self.password
        )

        self.realHandle = hkws_sdk.realPlay(self.userId)
        # hkws_sdk.setRealDataCallBack(userId, realHandle)
        # voiceHandle = hkws_sdk.startVoiceComMr(userId)

        hkws_sdk.ptzControlOther(self.userId, 1, hkws_sdk.DeviceCommand.PAN_LEFT, 0)

        asyncio.create_task(self.extractAudio())
        asyncio.create_task(self.readFrames())
        asyncio.create_task(self.handleFrames())
        asyncio.create_task(self.pushFrames())

    async def initBack(self):
        await super().initBack()
        await self.audioSource.subscribe(self.main.audioComponent.channelPlays[0])

    async def release(self):
        await super().release()
        if self.realHandle != -1:
            hkws_sdk.stopPreview(self.realHandle)
        if self.voiceHandle != -1:
            hkws_sdk.stopVoiceCom(self.voiceHandle)
        if self.userId != -1:
            hkws_sdk.logout(self.userId)

    async def readFrames(self):

        while True:

            try:

                logger.info("尝试连接摄像头...")

                # gst_pipeline = f"rtspsrc location={pushRtspUrl} latency=0 ! rtph264depay ! h264parse ! nvh264dec ! videoconvert ! appsink"
                # cap = await asyncio.get_event_loop().run_in_executor(None, cv2.VideoCapture, gst_pipeline, cv2.CAP_GSTREAMER)

                self.cap = await asyncio.get_event_loop().run_in_executor(
                    None, cv2.VideoCapture, self.getCameraRtspUrl()
                )

                if not self.cap.isOpened():
                    logger.warning("连接失败，5秒后重试...")
                    await asyncio.sleep(5)
                    continue

                while True:
                    ret: bool
                    frame: cv2.typing.MatLike

                    ret, frame = await asyncio.get_event_loop().run_in_executor(
                        None, self.cap.read
                    )

                    if not ret:
                        logger.error("读取帧失败，释放资源并重连...")
                        raise Exception("读取帧失败...")

                    h, w = frame.shape[:2]
                    if w < h:
                        frame = await asyncio.get_event_loop().run_in_executor(
                            None, cv2.rotate, frame, cv2.ROTATE_90_CLOCKWISE
                        )

                    await self.source.publish(frame)

            except asyncio.CancelledError:
                logger.info("任务被取消，执行清理...")
                await self.releaseCap()
                raise

            except Exception as e:
                logger.exception(f"发生未处理异常:  {str(e)}")
                await self.releaseCap()
                logger.info("5秒后尝试重新连接摄像头...")
                await asyncio.sleep(5)

            pass

    async def extractAudio(self):

        ffmpeg_command = [
            "ffmpeg",
            "-i",
            self.getCameraRtspUrl(),  # 输入RTSP流
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
            await self.audioSource.publish(b)

        await ByteFFmpegPull(
            ffmpeg_command, BUFFER_SIZE, publishToAudio, "cameraAudio"
        ).loop()

        pass

    def getCameraRtspUrl(self) -> str:
        return util.fillStr(
            self.cameraRtspUrl,
            {
                "username": self.username,
                "password": self.password,
                "ip": self.ip,
                "rtspPort": self.rtspPort,
            },
        )

    async def pushFrames(self):
        await FFmpegPushFrame(
            self.width,
            self.height,
            self.fps,
            self.pushRtspUrl,
            await self.out.subscribe(asyncio.Queue(maxsize=16)),
            __name__,
        ).loop()
        pass

    async def handleFrames(self):

        framesQueue: asyncio.Queue[cv2.typing.MatLike] = await self.source.subscribe(
            asyncio.Queue(maxsize=16)
        )

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
                        await self.identifyKeyframe.publish(res)

                    def _runDetection(
                        inputImage: cv2.typing.MatLike, useModel: list[detection.Model]
                    ):
                        # start_time = time.perf_counter()
                        res = self.main.detectionComponent.runDetection(
                            inputImage, useModel
                        )
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
                await self.out.publish(await _res.drawOutputImageAsunc())

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception(f"处理帧时发生异常: {str(e)}")

                res = None
                if task is not None:
                    task.cancel()
                    task = None
            pass

    async def releaseCap(self):
        if self.cap and self.cap.isOpened():
            await asyncio.get_event_loop().run_in_executor(None, self.cap.release)
        self.cap = None  # type: ignore
