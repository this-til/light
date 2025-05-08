import pygame
import logging
import asyncio
import audioop
import camera
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
    await camera.audioSource.subscribe(out)

    asyncio.create_task(playStreamLoop())


async def releaseAudio():
    """Release the audio system."""
    pygame.mixer.quit()
    pass
