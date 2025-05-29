#!/usr/bin/python3

import cv2
import asyncio
import logging
import numpy as np
import util
import detection
from util import Broadcaster, FFmpegPushFrame
from pyorbbecsdk import *
from main import Component, ConfigField


class OrbbecCameraComponent(Component):
    
    url: ConfigField[str] = ConfigField()
    width: ConfigField[int] = ConfigField()
    height: ConfigField[int] = ConfigField()
    fps: ConfigField[int] = ConfigField()
    pushRtspUrl: ConfigField[str] = ConfigField()

    config: Config | None = None
    pipeline: Pipeline | None = None

    source: Broadcaster[cv2.typing.MatLike] = Broadcaster()
    identifyKeyframe: Broadcaster[detection.Result] = Broadcaster()

    async def init(self):
        await super().init()
        asyncio.create_task(self.readImageLoop())
        asyncio.create_task(self.handleFrames())
        asyncio.create_task(self.pushFrames())

    async def readImageLoop(self):

        global config
        global pipeline

        config = None
        pipeline = None

        while True:
            try:

                config = Config()
                pipeline = Pipeline()

                profile_list = pipeline.get_stream_profile_list(
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

                config.enable_stream(color_profile)
                pipeline.start(config)

                while True:

                    frames: FrameSet = await asyncio.get_event_loop().run_in_executor(
                        None, pipeline.wait_for_frames, int(1000 / self.fps)
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
                if pipeline is not None:
                    pipeline.stop()
                raise
            except Exception as e:
                self.logger.exception(f"发生未处理异常:  {str(e)}")
                if pipeline is not None:
                    pipeline.stop()
                self.logger.info("5秒后尝试重新连接摄像头...")
                await asyncio.sleep(5)
            pass

    async def handleFrames(self):

        framesQueue: asyncio.Queue[cv2.typing.MatLike] = await self.source.subscribe(
            asyncio.Queue(maxsize=1)
        )

        while True:
            try:
               frame = await framesQueue.get()
               
               await asyncio.get_event_loop().run_in_executor(
                   None, self.main.detectionComponent.runDetection, frame, [detection.faceModel]
               )
               
               await asyncio.sleep(3)  
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(f"处理帧时发生异常: {str(e)}")
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
      