import pygame
import logging
import asyncio
import audioop
import camera
from camera import CameraVoiceData
from typing import *
from util import CircularBuffer

logger = logging.getLogger(__name__)

out: asyncio.Queue[bytes] = asyncio.Queue(maxsize=16)
channel: pygame.mixer.Channel = None  # type: ignore


async def playStreamLoop():

    while True:
        try:
            buf = await out.get()
            sound = pygame.mixer.Sound(buffer=buf)
            channel.queue(sound)
        except asyncio.CancelledError:
            channel.stop()
            raise
        except Exception as e:
            logger.exception(f"音频播放的未知异常:{str(e) }")


async def readBytesStreamLoop():

    queue: asyncio.Queue[CameraVoiceData] = asyncio.Queue(maxsize=16)

    await camera.camera.voiceBroadcaster.subscribe(queue)

    while True:
        try:
            data: CameraVoiceData = await queue.get()
            await out.put(data.data)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception(f"音频读取的未知异常:{str(e) }")


async def initAudio():
    pygame.mixer.init(
        frequency=camera.FREQUENCY,
        size=camera.SAMPLE_SIZE,
        channels=camera.CHANNELS,
        buffer=camera.SAMPLE_SIZE,
    )
    pygame.mixer.init()
    if pygame.mixer.get_init() is None:
        logger.error("Failed to initialize the audio system.")
        return

    global channel
    channel = pygame.mixer.Channel(0)

    pygame.mixer.music.set_volume(1)
    # await camera.audioSource.subscribe(out)

    asyncio.create_task(readBytesStreamLoop())
    asyncio.create_task(playStreamLoop())


async def releaseAudio():
    """Release the audio system."""
    pygame.mixer.quit()
    pass
