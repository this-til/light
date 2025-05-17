import pygame
import logging
import asyncio
import audioop
import camera
import hkws_sdk
from hkws_sdk import CameraVoiceData
from typing import *
from main import Component, ConfigField


class AudioComponent(Component):
    frequency: ConfigField[int] = ConfigField()
    size: ConfigField[int] = ConfigField()
    channelNumber: ConfigField[int] = ConfigField()
    buffer: ConfigField[int] = ConfigField()

    channelPlays: list[asyncio.Queue[bytes]] = []
    channels: list[pygame.mixer.Channel] = []

    async def init(self):
        await super().init()

        pygame.mixer.init(
            frequency=camera.FREQUENCY,
            size=camera.SAMPLE_SIZE,
            channels=camera.CHANNELS,
            buffer=camera.BUFFER_SIZE,
        )

        pygame.mixer.init()

        if pygame.mixer.get_init() is None:
            self.logger.error("Failed to initialize the audio system.")
            return

        pygame.mixer.music.set_volume(1)

        for i in range(self.channelNumber):
            queue = asyncio.Queue(maxsize=16)
            channel = pygame.mixer.Channel(i)
            self.channelPlays.append(queue)
            self.channels.append(channel)
            asyncio.create_task(self.playStreamLoop(queue, channel))
            
            
    async def release(self): 
        await super().release()
        for channel in self.channels:
            channel.stop()
        pygame.mixer.quit()

    async def playStreamLoop(
        self, queue: asyncio.Queue[bytes], channel: pygame.mixer.Channel
    ):

        while True:
            try:
                buf = await queue.get()
                sound = pygame.mixer.Sound(buffer=buf)
                channel.queue(sound)
            except asyncio.CancelledError:
                channel.stop()
                raise
            except Exception as e:
                self.logger.exception(f"音频播放的未知异常:{str(e) }")

    def getChannel(self, index: int) -> pygame.mixer.Channel | None:
        if index < 0 or index >= len(self.channels):
            return None
        return self.channels[index]
