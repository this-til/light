#!/usr/bin/python3

import logging
import asyncio
import serial_asyncio
from concurrent.futures import ThreadPoolExecutor

from serial import Serial
from util import Broadcaster
from main import Component, ConfigField

# 创建专用的串口线程池
uart_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="UART")


# class UartComponent(Component):
#
#    url : ConfigField[str] = ConfigField()
#    baudrate : ConfigField[int] = ConfigField()
#    bytesize : ConfigField[int] = ConfigField()
#    parity : ConfigField[str] = ConfigField()
#    stopbits  : ConfigField[int] = ConfigField()
#
#    reader: asyncio.StreamReader = None  # type: ignore
#    writer: asyncio.StreamWriter = None  # type: ignore
#
#    usarReader: Broadcaster[bytes] = Broadcaster()
#
#    async def init(self):
#        await super().init()
#        self.reader, self.writer = await serial_asyncio.open_serial_connection(
#            url=self.url,
#            baudrate=self.baudrate,
#            bytesize=self.bytesize,
#            parity=self.parity,
#            stopbits=self.stopbits,
#        )
#        asyncio.create_task(self.asyncReadSerialLoop())
#    pass
#
#    async def release(self):
#        await super().release()
#        if self.writer and not self.writer.is_closing():
#            self.writer.close()
#            await self.writer.wait_closed()
#
#    async def asyncReadSerialLoop(self):
#        while True:
#            try:
#                data = await self.reader.read(4096)
#                await self.usarReader.publish(data)
#            except Exception as e:
#                self.logger.exception(f"Error reading from serial: {str(e)}")
#
#


class UartComponent(Component):

    url: ConfigField[str] = ConfigField()
    baudrate: ConfigField[int] = ConfigField()
    bytesize: ConfigField[int] = ConfigField()
    parity: ConfigField[str] = ConfigField()
    stopbits: ConfigField[int] = ConfigField()

    usarReader: Broadcaster[bytes] = Broadcaster()

    uart: Serial = None  # type: ignore

    async def init(self):
        await super().init()
        self.uart = Serial(
            self.url,
            baudrate=self.baudrate,
            bytesize=self.bytesize,
            stopbits=self.stopbits,
            parity=self.parity,
            timeout=1,
        )
        asyncio.create_task(self.readLoop())

    async def readLoop(self):
        while True:
            try:
                data = await asyncio.get_event_loop().run_in_executor(
                    uart_executor, self.uart.read, 4096
                )

                if len(data) == 0:
                    continue

                await self.usarReader.publish(data)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(f"Error reading from serial: {str(e)}")
                await asyncio.sleep(5)

    def writeAsync(self, data: bytes):
        #self.uart.write(data)
        return asyncio.get_event_loop().run_in_executor(
            uart_executor, self.uart.write, data
        )
    
    async def release(self):
        """清理资源"""
        await super().release()
        
        # 关闭串口
        if self.uart and self.uart.is_open:
            try:
                self.uart.close()
            except Exception as e:
                self.logger.error(f"Error closing UART: {e}")
                
        # 关闭线程池
        if 'uart_executor' in globals():
            uart_executor.shutdown(wait=True)
