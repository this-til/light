#!/usr/bin/python3

import asyncio
import logging
import time

import cv2

import detection
import hkws_sdk
import util
from hkws_sdk import DeviceCommand
from main import Component, ConfigField
from util import Broadcaster, FFmpegPushFrame, ByteFFmpegPull
from typing import Sequence
from command import CommandEvent

# 音频参数
FREQUENCY = 16000  # 采样率16kHz
BITRATE = "32k"
SAMPLE_SIZE = -16  # 16位有符号PCM
CHANNELS = 1  # 单声道
BUFFER_SIZE = 16384  # 每次读取的数据块大小，须为2的倍数

ptzControlMap: dict[str, hkws_sdk.DeviceCommand] = {
    "TILT_UP": hkws_sdk.DeviceCommand.TILT_UP,
    "TILT_DOWN": hkws_sdk.DeviceCommand.TILT_DOWN,
    "PAN_LEFT": hkws_sdk.DeviceCommand.PAN_LEFT,
    "PAN_RIGHT": hkws_sdk.DeviceCommand.PAN_RIGHT,
}


class CameraComponent(Component):
    enable: ConfigField[bool] = ConfigField()
    enablePushFrames: ConfigField[bool] = ConfigField()
    enableDetection: ConfigField[bool] = ConfigField()

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
    # out: Broadcaster[cv2.typing.MatLike] = Broadcaster()
    detectionKeyframe: Broadcaster[detection.Result] = Broadcaster()

    cap: cv2.VideoCapture = None  # type: ignore

    sustainedDetection = False
    sustainedDetectionModel: detection.Model | None = None
    sustainedDetectionCondition = asyncio.Condition()

    sustainedDetectionKeyframe: Broadcaster[detection.Result] = Broadcaster()

    userId: int = -1
    realHandle: int = -1
    voiceHandle: int = -1

    ptzControlTask: asyncio.Task | None = None

    async def init(self):
        self.userId = hkws_sdk.login(
            self.ip, self.sdkPort, self.username, self.password
        )

        # self.realHandle = hkws_sdk.realPlay(self.userId)
        # hkws_sdk.ptzControlOther(self.userId, 1, hkws_sdk.DeviceCommand.PAN_LEFT, 0)

        if self.enable:
            asyncio.create_task(self.extractAudio())
            asyncio.create_task(self.readFrames())
            asyncio.create_task(self.commandEventHandle())

            if self.enableDetection:
                asyncio.create_task(self.handleFrames())
                asyncio.create_task(self.sustainedHandleFrames())

            if self.enablePushFrames:
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

                self.logger.info("尝试连接摄像头...")

                # gst_pipeline = f"rtspsrc location={pushRtspUrl} latency=0 ! rtph264depay ! h264parse ! nvh264dec ! videoconvert ! appsink"
                # cap = await asyncio.get_event_loop().run_in_executor(None, cv2.VideoCapture, gst_pipeline, cv2.CAP_GSTREAMER)

                self.cap = await asyncio.get_event_loop().run_in_executor(
                    None, cv2.VideoCapture, self.getCameraRtspUrl()
                )

                if not self.cap.isOpened():
                    self.logger.warning("连接失败，5秒后重试...")
                    await asyncio.sleep(5)
                    continue

                while True:
                    ret: bool
                    frame: cv2.typing.MatLike

                    ret, frame = await asyncio.get_event_loop().run_in_executor(
                        None, self.cap.read
                    )

                    if not ret:
                        self.logger.error("读取帧失败，释放资源并重连...")
                        raise Exception("读取帧失败...")

                    h, w = frame.shape[:2]
                    if w < h:
                        frame = await asyncio.get_event_loop().run_in_executor(
                            None, cv2.rotate, frame, cv2.ROTATE_90_CLOCKWISE
                        )

                    await self.source.publish(frame)

            except asyncio.CancelledError:
                self.logger.info("任务被取消，执行清理...")
                await self.releaseCap()
                raise

            except Exception as e:
                self.logger.exception(f"发生未处理异常:  {str(e)}")
                await self.releaseCap()
                self.logger.info("5秒后尝试重新连接摄像头...")
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
            "-b:a",
            BITRATE,
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
            await self.source.subscribe(asyncio.Queue(maxsize=16)),
            __name__,
        ).loop()
        pass

    async def handleFrames(self):
        framesQueue: asyncio.Queue[cv2.typing.MatLike] = await self.source.subscribe(
            asyncio.Queue(maxsize=1)
        )

        while True:
            try:
                async with self.sustainedDetectionCondition:
                    while self.sustainedDetection:
                        await self.sustainedDetectionCondition.wait()

                    sourceFrame: cv2.typing.MatLike = await framesQueue.get()
                    res: detection.Result = await self.main.detectionComponent.runDetection(
                        sourceFrame, self.main.detectionComponent.modelList
                    )

                    # Remove standing person results from fall detection
                    if self.main.detectionComponent.fallDownModel in res.cellMap:
                        cells = res.cellMap[self.main.detectionComponent.fallDownModel]
                        filtered_cells = [
                            cell for cell in cells
                            if cell.item != detection.FallDownModel.standPerson
                        ]
                        res.cellMap[self.main.detectionComponent.fallDownModel] = filtered_cells

                    await self.detectionKeyframe.publish(res)
                    await asyncio.sleep(3)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(f"处理帧时发生异常: {str(e)}")
                pass

    async def sustainedHandleFrames(self):
        framesQueue: asyncio.Queue[cv2.typing.MatLike] = await self.source.subscribe(
            asyncio.Queue(maxsize=1)
        )

        while True:
            try:
                async with self.sustainedDetectionCondition:
                    while not self.sustainedDetection:
                        await self.sustainedDetectionCondition.wait()

                    sourceFrame: cv2.typing.MatLike = await framesQueue.get()

                    if self.sustainedDetectionModel is None:
                        continue

                    res: detection.Result = await self.main.detectionComponent.runDetection(
                        sourceFrame, [self.sustainedDetectionModel]
                    )
                    await self.sustainedDetectionKeyframe.publish(res)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(f"处理帧时发生异常: {str(e)}")
                pass

    async def ptzControl(self, command: hkws_sdk.DeviceCommand):
        try:
            hkws_sdk.ptzControlOther(self.userId, 1, command, 0)
            await asyncio.sleep(1)
        finally:
            hkws_sdk.ptzControlOther(self.userId, 1, command, 1)
            self.ptzControlTask = None

        pass

    async def commandEventHandle(self):
        queue: asyncio.Queue[CommandEvent] = await self.main.commandComponent.commandEvent.subscribe(
            asyncio.Queue(maxsize=8))
        while True:
            try:
                event: CommandEvent = await queue.get()
                if event.key == "Detection.Sustained":
                    async with self.sustainedDetectionCondition:
                        if event.value in self.main.detectionComponent.modelMap:
                            self.sustainedDetectionModel = self.main.detectionComponent.modelMap[event.value]
                        else:
                            self.sustainedDetectionModel = None

                        self.sustainedDetection = self.sustainedDetectionModel is not None
                        self.sustainedDetectionCondition.notify_all()

                if event.key == "Camera.PtzControl":
                    if event.value not in ptzControlMap:
                        continue

                    deviceCommand : DeviceCommand = ptzControlMap[event.value]

                    if self.ptzControlTask is not None:
                        self.ptzControlTask.cancel()
                        self.ptzControlTask = None

                    self.ptzControlTask = asyncio.create_task(self.ptzControl(deviceCommand))


            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(f"处理命令时发生异常: {str(e)}")
                pass
        pass

    #    async def handleFrames(self):
    #
    #        framesQueue: asyncio.Queue[cv2.typing.MatLike] = await self.source.subscribe(
    #            asyncio.Queue(maxsize=16)
    #        )
    #
    #        res: detection.Result | None = None
    #        task: asyncio.Future[detection.Result] | None = None
    #
    #        while True:
    #            try:
    #
    #                sourceFrame: cv2.typing.MatLike = await framesQueue.get()
    #
    #                if res is None:
    #                    res = detection.Result(sourceFrame, {})
    #
    #                if task is None or task.done():
    #
    #                    if task is not None:
    #                        res = task.result()
    #                        await self.identifyKeyframe.publish(res)
    #
    #                    task = asyncio.get_event_loop().run_in_executor(
    #                        None,
    #                        self.main.detectionComponent.runDetection,
    #                        sourceFrame,
    #                        [self.main.detectionComponent.carAccidentModel],
    #                    )
    #
    #                _res = detection.Result(sourceFrame, res.cellMap)
    #                await self.out.publish(await _res.drawOutputImageAsync())
    #
    #            except asyncio.CancelledError:
    #                raise
    #            except Exception as e:
    #                logger.exception(f"处理帧时发生异常: {str(e)}")
    #
    #                res = None
    #                if task is not None:
    #                    task.cancel()
    #                    task = None
    #            pass

    async def releaseCap(self):
        if self.cap and self.cap.isOpened():
            await asyncio.get_event_loop().run_in_executor(None, self.cap.release)
        self.cap = None  # type: ignore
