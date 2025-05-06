#!/usr/bin/python3

import cv2
import asyncio
import logging
import numpy as np
import util
from util import Broadcaster, FFmpegPush
from pyorbbecsdk import *

logger = logging.getLogger(__name__)

width, height = 640, 480
fps = 30
pushRtspUrl = "rtsp://localhost:8554/stream2"


config: Config | None = None
pipeline: Pipeline | None = None

source: Broadcaster[cv2.typing.MatLike] = Broadcaster()
out: Broadcaster[cv2.typing.MatLike] = Broadcaster()


async def readImageLoop():

    global config
    global pipeline

    config = None
    pipeline = None

    while True:
        try:

            config = Config()
            pipeline = Pipeline()

            profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)

            try:

                color_profile: VideoStreamProfile = (
                    profile_list.get_video_stream_profile(
                        width, height, OBFormat.RGB, fps
                    )
                )

            except OBError as e:
                logger.exception(f"custom configuration error: {str(e)}")
                logger.info("use default configuration")
                color_profile = profile_list.get_default_video_stream_profile()
                logger.info(f"color profile: {str(color_profile)}")

            config.enable_stream(color_profile)
            pipeline.start(config)

            while True:

                frames: FrameSet = await asyncio.get_event_loop().run_in_executor(
                    None, pipeline.wait_for_frames, int(1000 / fps)
                )

                if frames is None:
                    continue

                color_frame = frames.get_color_frame()
                if color_frame is None:
                    continue

                color_image = util.frame_to_bgr_image(color_frame)
                if color_image is None:
                    logger.warning("failed to convert frame to image")
                    continue

                await source.publish(color_image)

                pass
        except asyncio.CancelledError:
            if pipeline is not None:
                pipeline.stop()
            raise
        except Exception as e:
            logger.exception(f"发生未处理异常:  {str(e)}")
            if pipeline is not None:
                pipeline.stop()
            logger.info("5秒后尝试重新连接摄像头...")
            await asyncio.sleep(5)
        pass


async def handleFrames():

    framesQueue: asyncio.Queue[cv2.typing.MatLike] = asyncio.Queue(maxsize=16)
    await source.subscribe(framesQueue)

    while True:
        try:

            sourceFrame = await framesQueue.get()
            await out.publish(sourceFrame)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception(f"处理帧时发生异常: {str(e)}")
        pass


async def pushFrames():
    await FFmpegPush(
        width,
        height,
        fps,
        pushRtspUrl,
        await source.subscribe(asyncio.Queue(maxsize=16)),
        __name__,
    ).pushFrames()
    pass


async def initOrbbecCamera():
    asyncio.create_task(readImageLoop())
    asyncio.create_task(handleFrames())
    asyncio.create_task(pushFrames())
    pass


async def releaseOrbbecCamera():

    pass
