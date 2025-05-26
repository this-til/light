import pyaudio
import wave
import util
import asyncio
from util import Broadcaster
from main import Component, ConfigField


class MicrophoneComponent(Component):

    framesPerBuffer: ConfigField[int] = ConfigField()
    rate: ConfigField[int] = ConfigField()
    channels: ConfigField[int] = ConfigField()

    pyAudio: pyaudio.PyAudio = None  # type: ignore
    stream = None  # type: ignore
    collectSound: Broadcaster[bytes] = Broadcaster()

    async def init(self):
        await super().init()
        
        asyncio.create_task(self.collectSoundLoop())

    async def release(self):
        await super().release()
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        if self.pyAudio is not None:
            self.pyAudio.terminate()

    async def collectSoundLoop(self):

        self.pyAudio = pyaudio.PyAudio()

        self.stream = self.pyAudio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.framesPerBuffer,
        )

        while True:
            try:
                data = await asyncio.get_event_loop().run_in_executor(
                    None, self.stream.read, self.framesPerBuffer
                )
                await self.collectSound.publish(data)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.error(f"Error in collectSoundLoop: {e}")

    pass
