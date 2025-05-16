#!/usr/bin/python3

import logging
import asyncio
import serial_asyncio

from util import Broadcaster
from main import Component , ConfigField


class UartComponent(Component):
    
    url : ConfigField[str] = ConfigField()
    baudrate : ConfigField[int] = ConfigField()
    bytesize : ConfigField[int] = ConfigField()
    parity : ConfigField[str] = ConfigField()
    stopbits  : ConfigField[int] = ConfigField()

    reader: asyncio.StreamReader = None  # type: ignore
    writer: asyncio.StreamWriter = None  # type: ignore
    
    usarReader: Broadcaster[bytes] = Broadcaster()

    async def init(self):
        await super().init()
        self.reader, self.writer = await serial_asyncio.open_serial_connection(
            url=self.url,
            baudrate=self.baudrate,
            bytesize=self.bytesize,
            parity=self.parity,
            stopbits=self.stopbits,
        )
        asyncio.create_task(self.asyncReadSerialLoop())
    pass

    async def release(self):
        await super().release()
        if self.writer and not self.writer.is_closing():
            self.writer.close()
            await self.writer.wait_closed()
            
    async def asyncReadSerialLoop(self):
        while True:
            try:
                data = await self.reader.read(1024)
                await self.usarReader.publish(data)
            except Exception as e:
                self.logger.exception(f"Error reading from serial: {str(e)}")   
                
