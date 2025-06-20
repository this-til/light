#!/usr/bin/python3
from __future__ import annotations

import asyncio

import cv2
from pyorbbecsdk import *

import detection
import util
from command import CommandEvent
from main import Component, ConfigField
from util import Broadcaster, FFmpegPushFrame


class OrbbecCameraComponent(Component):
    enable: ConfigField[bool] = ConfigField()
    enablePushFrames: ConfigField[bool] = ConfigField()

    width: ConfigField[int] = ConfigField()
    height: ConfigField[int] = ConfigField()
    fps: ConfigField[int] = ConfigField()
    pushRtspUrl: ConfigField[str] = ConfigField()

    config: Config | None = None
    pipeline: Pipeline | None = None

    sustainedDetection = False
    sustainedDetectionModel: detection.Model | None = None
    sustainedDetectionCondition = asyncio.Condition()

    source: Broadcaster[cv2.typing.MatLike] = Broadcaster()
    detectionKeyframe: Broadcaster[detection.Result] = Broadcaster()
    sustainedDetectionKeyframe: Broadcaster[detection.Result] = Broadcaster()

    async def init(self):
        await super().init()
        if self.enable:
            asyncio.create_task(self.readImageLoop())
            # asyncio.create_task(self.handleFrames())

            if self.enablePushFrames:
                asyncio.create_task(self.pushFrames())

    async def readImageLoop(self):

        while True:
            try:

                self.config = Config()
                self.pipeline = Pipeline()

                profile_list = self.pipeline.get_stream_profile_list(
                    OBSensorType.COLOR_SENSOR
                )

                try:

                    color_profile: VideoStreamProfile = (
                        profile_list.get_video_stream_profile(
                            self.width, self.height, OBFormat.RGB, self.fps
                        )
                    )

                except OBError as e:
                    self.logger.exception(f"custom configuration error: {str(e)}")
                    self.logger.info("use default configuration")
                    color_profile = profile_list.get_default_video_stream_profile()
                    self.logger.info(f"color profile: {str(color_profile)}")

                self.config.enable_stream(color_profile)
                self.pipeline.start(self.config)

                while True:

                    frames: FrameSet = await asyncio.get_event_loop().run_in_executor(
                        None, self.pipeline.wait_for_frames, int(1000 / self.fps)
                    )

                    if frames is None:
                        continue

                    color_frame = frames.get_color_frame()
                    if color_frame is None:
                        continue

                    color_image = await asyncio.get_event_loop().run_in_executor(
                        None, util.frame_to_bgr_image, color_frame
                    )
                    if color_image is None:
                        self.logger.warning("failed to convert frame to image")
                        continue

                    await self.source.publish(color_image)

                    pass

            except asyncio.CancelledError:
                if self.pipeline is not None:
                    self.pipeline.stop()
                raise
            except Exception as e:
                self.logger.exception(f"发生未处理异常:  {str(e)}")
                if self.pipeline is not None:
                    self.pipeline.stop()
                self.logger.info("5秒后尝试重新连接摄像头...")
                await asyncio.sleep(5)
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

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(f"处理命令时发生异常: {str(e)}")
                pass

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
